import torch
from torch import nn


class FashionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden_layer1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.hidden_layer2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.hidden_layer1(x))
        x = self.relu2(self.hidden_layer2(x))
        return self.output_layer(x)
