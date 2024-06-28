import pyRDDLGym
from pyRDDLGym.core.policy import RandomAgent

domainPath = 'RDDL/Domain0.rddl'
instancePath = 'RDDL/Instance0.rddl'

myEnv = pyRDDLGym.make(domain=domainPath, instance=instancePath)

agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.max_allowed_actions, seed=42)

total_reward = 0
state, _ = myEnv.reset()
for step in range(myEnv.horizon):
    myEnv.render(to_display=False)
    action = agent.sample_action()
    next_state, reward, done, info, _ = myEnv.step(action)
    total_reward += reward
    print()
    print(f'step       = {step}')
    print(f'state      = {state}')
    print(f'action     = {action}')
    print(f'next state = {next_state}')
    print(f'reward     = {reward}')
    state = next_state
    if done:
        break
print(f'episode ended with reward {total_reward}')

myEnv.close()