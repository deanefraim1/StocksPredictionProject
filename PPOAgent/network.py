"""
    This file contains a neural network module for us to
    define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import helpers

class FeedForwardNN(nn.Module):
    """
        A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """
    def __init__(self, in_dim, out_dim):
        """
            Initialize the network and set up the layers.

            Parameters:
                in_dim - input dimensions as an int
                out_dim - output dimensions as an int

            Return:
                None
        """
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.norm1 = nn.LayerNorm(64)  # Normalization layer after first linear layer

        self.layer2 = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)  # Normalization layer after second linear layer

        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        """
            Runs a forward pass on the neural network.

            Parameters:
                obs - observation to pass as input

            Return:
                output - the output of our forward pass
        """
        # First hidden layer: Linear -> Norm -> ReLU
        x = self.layer1(obs)
        x = self.norm1(x)
        x = F.relu(x)

        # Second hidden layer: Linear -> Norm -> ReLU
        x = self.layer2(x)
        x = self.norm2(x)
        x = F.relu(x)

        # Final layer: Linear -> Sigmoid -> Scale output
        x = self.layer3(x)
        output = 10 * torch.sigmoid(x)

        return output