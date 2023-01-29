import torch
import torchvision.models as models
from typing import Callable

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
        self.classifier = torch.nn.Linear(2048, 620)

        # Feature extraction
        self.layers = ["layer2", "layer3"]
        self._features = {layer: torch.empty(0) for layer in self.layers}

        for layer_id in self.layers:
            layer = dict([*self.model_resnet.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    def forward(self, img, word_emb):
        # low level feature extraction
        # use the output feature maps from ResNet block 2 and 3 as low-level features.
        # layer2.3.relu_2 -> torch.Size([1, 512, 28, 28])
        # layer3.5.relu_2 -> torch.Size([1, 1024, 14, 14])
        # Low feature = dict_keys(['layer2', 'layer3'])

        _ = self.model_resnet(img)

        with torch.no_grad():
            layer2 = self._features['layer2'].clone()
            layer3 = self._features['layer3'].clone()

        img = self.model_resnet(img)
        word_emb = self.f_gate(word_emb)

        # channel like product of fgate anf fimg
        combination = torch.mul(img, word_emb)  # torch.Size([1, 2048])

        # concatenation of penultimate layer of Resnet and result of the compound module
        low_features = torch.cat((layer2, layer3.resize_(layer2.size())), dim=1)  # torch.Size([1, 1024, 28, 28])
        concatenation = torch.cat((low_features.resize_(combination.size()), combination), dim=1)

        output = torch.sigmoid(self.classifier(concatenation))

        return output


if __name__ == "__main__":

    # shape is (batch, C, H, W)
    img = torch.rand(1, 3, 224, 224)
    # img = torch.rand(1,3,512,812)

    # shape is (batch, dim)
    # 100d for pretrained GloVe embeddings
    word = torch.rand(1, 100)

    model = Model()
    out = model(img, word)
