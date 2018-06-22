import torch.nn as nn


class DQN(nn.Module):
    """
    A simple Deep Q-Network with fully connected layers.
    """
    def __init__(self, input_dims, output_dims):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dims)
        )

    def forward(self, x):
        return self.layers(x)
