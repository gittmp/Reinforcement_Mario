# Example of Q-Learning deep reinforcement agent on GridWorld
# Source: https://app.getpocket.com/read/3100336624

# Imports
import matplotlib.pyplot as plt


class Agent:

    def q_update(self, state, action_id, reward, next_state, terminal):
        ...
        if terminal:  # if the successor state is terminal, the TD target = R
            target = reward
        else:  # otherwise, TD target = R + y * max( Q(S', a') )
            target = reward + self.gamma * max(self.q_table[next_state])

        # TD error = TD target - Q(S, A)
        td_error = target - self.q_table[state, action_id]

        # update all cells in q-table, discounted by the learning rate a
        # Q(S, A) = Q(S, A) + a * TD error
        self.q_table[state, action_id] = self.q_table[state, action_id] + self.alpha * td_error


# Function to train a RL agent through Q-Learning
def train_agent():
    # training initialisation
    num_episodes = 2000  # set number of episodes to train over
    agent = Agent()  # agent from class
    env = Grid()  # environment from class
    rewards = []  # array holding final observed reward in each episode

    # training loop
    for _ in range(num_episodes):  # loop through episode limit
        # episode initialisation
        state = env.reset()  # reset environment to get a new initial state
        episode_reward = 0  # counter of total reward for current episode

        # episode loop
        while True:
            # SARSA sampling
            action_id, action = agent.act(state)  # pick next action A'
            next_state, reward, terminal = env.step(action)  # sample successor state, reward, and terminal from env
            episode_reward += reward  # update total reward for episode

            # SARSAMAX Q-update
            agent.q_update(state, action_id, reward, next_state, terminal)  # update all Q-values using SARSAMAX
            state = next_state  # set successor state as current before progressing forward

            # if the successor state is terminal, end the episode
            if terminal:
                break

        # after episode terminates, append total generated reward to rewards array
        rewards.append(episode_reward)

    # after training is complete, plot the reward array showing the progression of the agents success
    plt.plot(rewards)
    plt.show()

    # return the latest generated policy
    return agent.best_policy()


print(train_agent())