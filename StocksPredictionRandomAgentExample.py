import pyRDDLGym
from pyRDDLGym.core.policy import RandomAgent, NoOpAgent

domainPath = 'RDDL/Domain.rddl'
instancePath = 'RDDL/Instance0.rddl'

myEnv = pyRDDLGym.make(domain=domainPath, instance=instancePath)

randomAgent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.max_allowed_actions, seed=42)

noOpAgent = NoOpAgent(action_space=myEnv.action_space, num_actions=myEnv.max_allowed_actions)
    
total_reward = 0
state, _ = myEnv.reset()
for step in range(myEnv.horizon):
    myEnv.render(to_display=False)
    action = randomAgent.sample_action(state)
    next_state, reward, done, info, _ = myEnv.step(action)
    total_reward += reward
    print()
    print(f'Step                            = {step}')
    print(f'Current Time                    = {next(key for key, value in state.items() if value)}')
    print(f'Current Time Shares Status      = {list(state.items())[-2:]}')
    print(f'action                          = {action}')
    print(f'Next Time                       = {next(key for key, value in next_state.items() if value)}')
    print(f'Next Time Shares Status         = {list(next_state.items())[-2:]}')
    print(f'reward                          = {reward}')
    print(f'total_reward                    = {total_reward}')
    state = next_state
    if done:
        break
print(f'episode ended with reward {total_reward}')

myEnv.close()