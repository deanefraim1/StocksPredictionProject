"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

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
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor
		obs = self.extract_features_as_tensor(obs)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output
	
	def extract_features_as_tensor(self, obs) -> torch.Tensor:
		"""
			Extracts features from the observation.

			Parameters:
				obs - observation to pass as input

			Return:
				features - the features extracted from the observation
		"""
		if(isinstance(obs, np.ndarray)):
			return torch.tensor(obs, dtype=torch.float32)
		
		if(isinstance(obs, torch.Tensor)):
			return obs

		# Extract time value
		time_value = None
		for key, value in obs.items():
			if key.startswith("CurrentTime___t") and value:
				time_value = int(key.split("___t")[1])
				break

		# Extract the last two numbers
		stocks = list(obs.values())[-2:]

		# Create the tensor
		result_tensor = torch.tensor([time_value] + stocks, dtype=torch.float32)

		return result_tensor

