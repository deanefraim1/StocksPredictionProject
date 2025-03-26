import numpy as np
import torch

class StateStacker:
    def __init__(self, stack_size=3, obs_shape=(2,)):
        self.stack_size = stack_size
        self.obs_shape = obs_shape
        # Pre-allocate a fixed buffer with zeros.
        self.buffer = np.zeros((stack_size, *obs_shape))
    
    def reset(self):
        # Reset the buffer to all zeros.
        self.buffer[:] = 0
        return self.get_state()
    
    def update(self, obs):
        # Shift the buffer and add the new observation at the end.
        self.buffer = np.roll(self.buffer, shift=-1, axis=0)
        self.buffer[-1] = obs
        return self.get_state()
    
    def get_state(self):
        # Flatten the buffer and return as a torch tensor.
        return torch.tensor(self.buffer.flatten(), dtype=torch.float)