"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py. I wrote this file to demonstrate
	that our trained policy exists independently of our learning algorithm,
	which resides in ppo.py. Thus, we can test our trained policy without 
	relying on ppo.py.
"""
from collections import OrderedDict
import numpy as np
import helpers
import matplotlib.pyplot as plt
import os

def _log_summary(ep_len, ep_ret):
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
		print(f"-------------------- Episode Results --------------------", flush=True)
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
	obs, _ = env.reset()
	done = False

	# number of timesteps so far
	t = 0

	# Logging data
	ep_len = 0            # episodic length
	ep_ret = 0            # episodic return

	# Containers for plotting:
	reward_history = []   # To store cumulative reward at each step
	actions_hist = None   # To collect action values per key (will become a dict of lists)

	while not done:
		t += 1

		# Render environment if specified, off by default
		if render:
			env.render()

		# Query deterministic action from policy and run it
		action = policy(helpers.TransformPyRddlStateToPPOState(obs)).detach().numpy()
		pyrddlActionDict = helpers.TransformPPOActionToPyRddlAction(env.action_space, action)
		nextObs, rew, terminated, truncated, _ = env.step(pyrddlActionDict)

		# Sum all episodic rewards as we go along
		ep_ret += rew

		reward_history.append(ep_ret)

		if actions_hist is None:
			actions_hist = {key: [] for key in pyrddlActionDict}
		for key in pyrddlActionDict:
			actions_hist[key].append(pyrddlActionDict[key])
		
		print()
		print(f'Step                            = {t}')
		print(f'Current Time                    = {next(key for key, value in obs.items() if value)}')
		print(f'Current Time Shares Status      = {list(obs.items())[-2:]}')
		print(f'action                          = {pyrddlActionDict}')
		print(f'Next Time                       = {next(key for key, value in nextObs.items() if value)}')
		print(f'Next Time Shares Status         = {list(nextObs.items())[-2:]}')
		print(f'reward                          = {rew}')
		print(f'total_reward                    = {ep_ret}')

		obs = nextObs
		done = terminated | truncated

	# 1. Plot the total (cumulative) reward over time.
	fig, ax1 = plt.subplots(figsize=(10, 5))
	ax1.plot(range(1, t + 1), reward_history, color='blue', label='Total Reward')
	ax1.set_xlabel('Step')
	ax1.set_ylabel('Total Reward', color='blue')
	ax1.tick_params(axis='y', labelcolor='blue')
	ax1.set_title('Total Reward over Time')
	ax1.legend()
	current_dir = os.path.dirname(os.path.abspath(__file__))
	resultsDir = os.path.join(current_dir, 'ResultsFiles', 'reward_plot.png')
	plt.savefig(resultsDir)
	plt.close(fig)

	# 2. Plot histograms for each action key.
	if actions_hist is not None:
		keys = list(actions_hist.keys())
		num_keys = len(keys)
		# Arrange histograms in a grid; for 4 keys, a 2x2 grid works well.
		fig, axs = plt.subplots(2, 2, figsize=(12, 10))
		axs = axs.flatten()  # To simplify the loop
		for i, key in enumerate(keys):
			axs[i].hist(actions_hist[key], bins=100, edgecolor='black', alpha=0.7)
			axs[i].set_title(f'Histogram for {key}')
			axs[i].set_xlabel('Value')
			axs[i].set_ylabel('Frequency')
		plt.tight_layout()
		current_dir = os.path.dirname(os.path.abspath(__file__))
		resultsDir = os.path.join(current_dir, 'ResultsFiles', 'action_histograms.png')
		plt.savefig(resultsDir)
		plt.close(fig)
		
	# Track episodic length
	ep_len = t

	# returns episodic length and return in this iteration
	return ep_len, ep_ret

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
	# Rollout with the policy and environment, and log the episode's data
	ep_len, ep_ret = rollout(policy, env, render)
	_log_summary(ep_len=ep_len, ep_ret=ep_ret)