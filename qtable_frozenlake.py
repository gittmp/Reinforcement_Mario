import gym
import numpy as np
import matplotlib.pyplot as plt

# load environment of Frozen Lake
# env = gym.make('Taxi-v3')
env = gym.make('FrozenLake-v0', is_slippery=False)

# implement tabular q-learning

# initialise Q-table matrix of states x actions with all 0s
Q = np.zeros([env.observation_space.n, env.action_space.n])

# set learning parameters and reward array
alpha = 0.8  # learning rate
gamma = 0.95  # discount factor
episode_rewards = []
num_episodes = 1000

# training

# loop over all episodes
for ep in range(num_episodes):
    # reset environment to start new episode and initialise variables
    state = env.reset()
    total_reward = 0
    terminal = False
    time_step = 0

    # loop through episode, generating new states and taking new actions
    while time_step < 99:
        time_step += 1

        # choose next action greedily from Q-table featuring some random noise
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1/(ep+1)))
        # print("Action: ", action, " Type: ", action.dtype)

        # sample successor state and reward from environment (discard metadata)
        successor, reward, terminal, _ = env.step(action)

        # render environment frame
        # if np.random.uniform(0, 1) > 0.95:
        #     print("Episode = " + str(ep) + " Time-step = " + str(time_step))
        #     env.render()

        # update Q-table with new Q-values using Bellman equation for that state-action pair - SARSA TD Learning
        # Q-target = R + maximal action according to current Q-table
        td_target = reward + gamma * np.max(Q[successor, :])
        td_error = td_target - Q[state, action]
        Q[state, action] = Q[state, action] + alpha * td_error

        # update total reward for the game, and set current state as the successor
        total_reward += reward
        state = successor

        # if we've reached the terminal state, end the episode
        if terminal:
            break

    # update the array of all episode final rewards with that of the completed episode (win = 1, loss = 0)
    episode_rewards.append(total_reward)

# RESULTS

print("\nTotal average score: " + str(sum(episode_rewards)/num_episodes), end='\n\n')

# print("Average score of each section of training: ")
# num_sections = 100

# section_size = np.max([int(num_episodes/num_sections), 1])
# averages = []
# for i in range(num_sections):
#     low = i*section_size
#     high = (i+1)*section_size
#     # print("Section: " + str(low) + " -> " + str(high))
#     avg = sum(episode_rewards[low:high]) / section_size
#     averages.append(avg)
#     # print("Average: " + str(avg), end='\n\n')
#
# # Plot the reward array showing the progression of the agents success
# plt.plot(averages)
# plt.show()

# print("Final Q-table: ")
# print(Q)
