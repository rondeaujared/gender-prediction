import torch
from torch import nn


class AgeNext:

    def __init__(self, weights):
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        model.fc = nn.Linear(model.fc.in_features, 101)
        model.load_state_dict(torch.load(weights))
        self.model = model

    def getModel(self):
        return self.model
