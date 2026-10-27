
import torch
from torch import nn


class DNNClassifier(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)
