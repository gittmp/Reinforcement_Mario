import QL2DQL_def as defs
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0', is_slippery=False)
# env = gym.make('Taxi-v3')
state_space = env.observation_space.n
agent = defs.Agent(state_space,
                   env.action_space.n,
                   target_update_freq=100,
                   discount_factor=0.95,
                   batch_size=32,
                   epsilon_anneal_rate=(1/5000),
                   ermb_init_size=5000)

num_eps = 5000
max_ep_length = 99
ep_rewards = []
training = True

for ep in range(num_eps):
    # reset environment
    state = env.reset()

    # reset agent's episode log
    agent.start_episode()
    time_step = 0
    total_reward = 0
    terminal = False
    reward = None

    # print("Starting episode...", end='\n\n')

    # start episode loop
    while time_step < max_ep_length and not terminal:
        time_step += 1

        # print("State = ", state)

        state = np.array([s == state for s in range(state_space)], dtype='float32')
        observation = {
            'state': state,
            'reward': reward
        }

        action = int(agent.step(observation, training))

        # env.render()
        # print()

        successor, reward, terminal, _ = env.step(action)
        total_reward += reward
        state = successor

    ep_rewards.append(total_reward)

print("\nTotal average score: " + str(sum(ep_rewards)/num_eps))

num_sections = 100
section_size = np.max([int(num_eps/num_sections), 1])
averages = []
for i in range(num_sections):
    low = i*section_size
    high = (i+1)*section_size
    # print("Section: " + str(low) + " -> " + str(high))
    avg = sum(ep_rewards[low:high]) / section_size
    averages.append(avg)
    # print("Average: " + str(avg), end='\n\n')

# Plot the reward array showing the progression of the agents success
plt.plot(averages)
plt.show()
