import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BreastCancerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # MUST be named `self.model` to match checkpoint
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)
