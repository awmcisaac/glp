import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Callable
import typing as t

class Model(torch.nn.Module):

    # 2048 = number of channels in the penultimate layer of resnet50
    # 300 = size of word embeddings
    # 620 = number of labels
    # 100d for pretrained GloVe embeddings

    def __init__(self, device):
        super(Model, self).__init__()
        self.device = device

        # ResNet backbone
        self.model_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)  # Initializing ResNet50 by pytorch
#        self.model_resnet.fc = torch.nn.Identity()  # Replace the fully connected layer with an identity

        # gating mechanism
        self.f_gate = torch.nn.Sequential(
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 2048),
            torch.nn.Sigmoid(),
        )

        # classifier - composition 2048 + layer2 512 + layer3 1024 = 3584
        self.classifier = torch.nn.Linear(3584, 620)

        # Feature extraction
        # self.layers = ["layer2", "layer3"]
        # self._features = {layer: torch.empty(0) for layer in self.layers}

        # use the output feature maps from ResNet block 2 and 3 as low-level features.
        # layer4 is output before final fully connected layer
        self.return_nodes = {
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4"
        }
        self.model_resnet = create_feature_extractor(self.model_resnet, return_nodes=self.return_nodes)

        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, img, word_emb):
        img = self.model_resnet(img)
        word_emb = self.f_gate(word_emb)

        # channel like product of fgate and fimg
        composition = torch.mul(self.adaptive_avg_pool(img['layer4']).squeeze(3).squeeze(2), word_emb)  # torch.Size([1, 2048])

        # channel-wise concatenation of penultimate layer of Resnet and result of the compound module
        layer2 = self.adaptive_avg_pool(img['layer2']).squeeze(3).squeeze(2)
        layer3 = self.adaptive_avg_pool(img['layer3']).squeeze(3).squeeze(2)
        concatenation = torch.cat(
            [layer2, layer3, composition], dim=1
        ).to(self.device) # torch.Size([1, 1024, 28, 28])
        
        output = self.classifier(concatenation)

        return output

class KDModel(Model):
    def __init__(self, teacher_model_weights: t.Dict, device):
        super(KDModel, self).__init__(device)
        self.teacher_model = Model(device)
        self.teacher_model.load_state_dict(teacher_model_weights)
        self.teacher_model.requires_grad_(False) # don't train

    def forward(self, img, word_emb):
        teacher_logits = self.teacher_model(img, word_emb)

        img = self.model_resnet(img)
        word_emb = self.f_gate(word_emb)

        composition = torch.mul(self.adaptive_avg_pool(img['layer4']).squeeze(3).squeeze(2), word_emb)
        layer2 = self.adaptive_avg_pool(img['layer2']).squeeze(3).squeeze(2)
        layer3 = self.adaptive_avg_pool(img['layer3']).squeeze(3).squeeze(2)
        concatenation = torch.cat(
            [layer2, layer3, composition], dim=1
        ).to(self.device)

        student_logits = self.classifier(concatenation)

        return teacher_logits, student_logits


if __name__ == "__main__":

    # shape is (batch, C, H, W)
    img = torch.rand(1, 3, 224, 224)

    # shape is (batch, dim)
    # 100d for pretrained GloVe embeddings
    word = torch.rand(1, 100)

    model = Model(device="cpu")
    out = model(img, word)
