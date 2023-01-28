import torch
import torchvision
from torchvision import transforms as T # functional utilities
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor


import torchvision.models as models

#### MODEL
# Composition module (2-layer MLP)
# ResNet50 backbone
# Final model (ResNet50 + Composition + Classifier)

class Model(torch.nn.Module):

    # 2048 = number of channels in the penultimate layer of resnet50
    # 300 = size of word embeddings
    # 620 = number of labels
    # 100d for pretrained GloVe embeddings

    def __init__(self):
        super(Model, self).__init__()

        # ResNel backbone
        self.model_resnet = models.resnet50(pretrained=True)  # Initializing ResNet50 by pytorch
        self.model_resnet.fc = torch.nn.Identity()  # Replace the fully connected layer with an identity

        # gating mechanism
        self.f_gate = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.Sigmoid(),
            torch.nn.Linear(100, 2048)
        )

        # classifier
        # dim = 28*28
        self.classifier = torch.nn.Linear(2048, 620)



    def forward(self, img, word_emb):

        img = self.model_resnet(img)
        word_emb = self.f_gate(word_emb)

        # channel like product of fgate anf fimg -> keepind dim of x1, it supports backprop
        # torch.Size([1, 2048, 1, 2048])
        combination = torch.mul(img, word_emb.unsqueeze(dim=-1).unsqueeze(dim=-1))

        # low level feature extraction
        # use the output feature maps from ResNet block 2 and 3 as low-level features.
        # layer2.3.relu_2 -> torch.Size([1, 512, 28, 28])
        # layer3.5.relu_2 -> torch.Size([1, 1024, 14, 14])

        return_nodes = {
            "layer2.3.relu_2": "layer2",
            "layer3.5.relu_2": "layer3"
        }
        model2 = create_feature_extractor(img, return_nodes=return_nodes)
        intermediate_outputs = model2(torch.rand(1, 3, 224, 224))
        low_features = torch.cat(intermediate_outputs['layer2'], intermediate_outputs['layer3']) # list of 2 tensors

        # concatenation of penultimate layer of Resnet and result of the compound module
        concatenation = torch.cat((low_features, combination), dim=0)

        output = torch.sigmoid(self.classification(concatenation))

        return output


if __name__ == "__main__":
