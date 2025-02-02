"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py. I wrote this file to demonstrate
	that our trained policy exists independently of our learning algorithm,
	which resides in ppo.py. Thus, we can test our trained policy without 
	relying on ppo.py.
"""
from collections import OrderedDict
import numpy as np

def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def rollout(policy, env, render):
	"""
		Returns a generator to roll out each episode given a trained policy and
		environment to test on. 

		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
			render - Specifies whether to render or not
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.

		Note:
			If you're unfamiliar with Python generators, check this out:
				https://wiki.python.org/moin/Generators
			If you're unfamiliar with Python "yield", check this out:
				https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	"""
	# Rollout until user kills process
	while True:
		obs, _ = env.reset()
		obs = extract_features_as_numpyArray(obs)
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return

		while not done:
			t += 1

			# Render environment if specified, off by default
			if render:
				env.render()

			# Query deterministic action from policy and run it
			action = policy(obs).detach().numpy()
			obs, rew, terminated, truncated, _ = env.step(numpyArray_to_action(env.action_space, action))
			obs = extract_features_as_numpyArray(obs)
			done = terminated | truncated

			# Sum all episodic rewards as we go along
			ep_ret += rew
			
		# Track episodic length
		ep_len = t

		# returns episodic length and return in this iteration
		yield ep_len, ep_ret

def numpyArray_to_action(action_space, action):
	"""
		Converts a numpy array to an action dictionary.

		Parameters:
			action - the numpy array to convert

		Return:
			action_dict - the action dictionary
	"""
	# Get the keys from the action space than zero the 0, and 3rd index and insert the action in the 1st and 2nd index
	action_keys = list(action_space.keys())
	action_values = [0, max(action[0], 0), max(action[1], 0), 0]
	
	# Create an OrderedDict by pairing keys and values
	action_dict = OrderedDict(zip(action_keys, action_values))
	
	return action_dict

def extract_features_as_numpyArray(obs: dict) -> np.ndarray:
	"""
		Extracts features from the observation.

		Parameters:
			obs - observation to pass as input

		Return:
			features - the features extracted from the observation
	"""
	# Extract time value
	time_value = None
	for key, value in obs.items():
		if key.startswith("CurrentTime___t") and value:
			time_value = int(key.split("___t")[1])
			break

	# Extract the last two numbers
	stocks = list(obs.values())[-2:]

	# Create the tensor of positive integers
	result_array = np.array([time_value] + stocks, dtype=np.float32)

	return result_array

def eval_policy(policy, env, render=False):
	"""
		The main function to evaluate our policy with. It will iterate a generator object
		"rollout", which will simulate each episode and return the most recent episode's
		length and return. We can then log it right after. And yes, eval_policy will run
		forever until you kill the process. 

		Parameters:
			policy - The trained policy to test, basically another name for our actor model
			env - The environment to test the policy on
			render - Whether we should render our episodes. False by default.

		Return:
			None

		NOTE: To learn more about generators, look at rollout's function description
	"""
	# Rollout with the policy and environment, and log each episode's data
	for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)