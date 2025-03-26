import torch
import numpy as np
from collections import OrderedDict

def TransformPyRddlStateToPPOState(obs: dict) -> torch.Tensor:
    # Extract time value
    time_value = None
    for key, value in obs.items():
        if key.startswith("CurrentTime___t") and value:
            time_value = int(key.split("___t")[1])
            break

    # Extract the last two numbers
    stocks = list(obs.values())[-2:]

    # Create the tensor
    result_tensor = torch.tensor(stocks, dtype=torch.float32)

    return result_tensor

def TransformPPOActionToPyRddlAction(pyRddlActionSpace, ppoAction):
    # Get the keys from the action space than zero the 0, and 3rd index and insert the action in the 1st and 2nd index
    action_keys = list(pyRddlActionSpace.keys())
    action_values = [0, ppoAction[0], ppoAction[1], 0]
    
    # Create an OrderedDict by pairing keys and values
    pyRddlAction = OrderedDict(zip(action_keys, action_values))
    
    return pyRddlAction