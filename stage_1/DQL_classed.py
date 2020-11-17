# imports
import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as funt
import torch.optim as optim
import matplotlib.pyplot as plt
import collections
import time
import datetime


class Buffer:
    def __init__(self, max_buffer_size):
        self.buffer = collections.deque(maxlen=max_buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, size):
        batch = random.sample(self.buffer, size)
        states, actions, rewards, successors, terminal_masks = [], [], [], [], []

        for experience in batch:
            s, a, r, succ, term = experience
            states.append(s)
            actions.append([a])
            rewards.append([r])
            successors.append(succ)
            terminal_masks.append([term])

        state_tensor = torch.tensor(states, dtype=torch.float)
        action_tensor = torch.tensor(actions)
        reward_tensor = torch.tensor(rewards)
        successor_tensor = torch.tensor(successors, dtype=torch.float)
        terminal_tensor = torch.tensor(terminal_masks)

        return state_tensor, action_tensor, reward_tensor, successor_tensor, terminal_tensor

    def size(self):
        return len(self.buffer)


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        self.output_size = output_size
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size, 256)
        self.l2 = nn.Linear(256, 84)
        self.l3 = nn.Linear(84, output_size)

    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), -1)
        inputs = funt.relu(self.l1(inputs))
        inputs = funt.relu(self.l2(inputs))
        outputs = self.l3(inputs)

        return outputs

    def step(self, inputs, eps):
        outputs = self.forward(inputs)

        expl_probability = random.random()
        if expl_probability <= eps:
            return random.randint(0, self.output_size - 1)
        else:
            return outputs.argmax().item()


class Agent:
    def __init__(self,
                 states_n,
                 actions_n,
                 alpha=0.005,
                 gamma=0.98,
                 max_buffer_size=50000,
                 batch_size=32,
                 min_epsilon=0.01,
                 max_epsilon=0.11,
                 annealing_rate=0.0005,
                 init_training=1000,
                 update_target_rate=20):

        # set hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.annealing_rate = annealing_rate
        self.init_training = init_training
        self.update_target = update_target_rate

        # create replay buffer and agent's two networks: optimising and target (init target params equal to optimising)
        self.optimising_network = Network(states_n, actions_n)
        self.target_network = Network(states_n, actions_n)
        self.replay_buffer = Buffer(max_buffer_size)
        self.target_network.load_state_dict(self.optimising_network.state_dict())
        self.optimiser = optim.Adam(self.optimising_network.parameters(), lr=alpha)

    def select_epsilon(self, ep_no):
        return max(self.min_epsilon, self.max_epsilon - self.annealing_rate * ep_no)

    def policy(self, s, e):
        return self.optimising_network.step(torch.from_numpy(s).float().unsqueeze(0), e)

    def add_experience(self, experience):
        self.replay_buffer.add(experience)

    def train_step(self, ep_no):
        if self.replay_buffer.size() > self.init_training:
            for i in range(10):
                s, a, r, succ, term = self.replay_buffer.sample(self.batch_size)

                qvalues = self.optimising_network(s)
                q_sa = qvalues.gather(1, a)

                # generate q values for all actions from the successor state
                succ_qvalues = self.target_network(succ)
                # generate tensor of the maximal action for the successor state for each element in the batch
                max_q_succ = succ_qvalues.max(1)[0].unsqueeze(1)
                # Q-target = reward + discounted maximal value of successor state (over all actions)
                # if terminal, just reward (mask out rest)
                q_target = r + self.gamma * max_q_succ * term

                loss = funt.smooth_l1_loss(q_sa, q_target)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

            # at regular intervals bring target network up to date with optimising network
            if ep_no % self.update_target == 0 and ep != 0:
                self.target_network.load_state_dict(self.optimising_network.state_dict())


# initialise environment
# env = gym.make('CartPole-v1')
env = gym.make('Pong-v0')
# env = gym.make('FrozenLake-v0', is_slippery=False)

print("State space shape: ", env.observation_space.shape)
# print("State space high: ", env.observation_space.high)
# print("State space low: ", env.observation_space.low)
print("Action space: ", env.action_space)
start = time.time()

breakpoint()

# seed behaviour of spaces such that they are reproducible
seed = 742
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
env.seed(seed)
env.action_space.seed(seed)

state_space_size = np.array(env.observation_space.shape).prod()
action_space_size = env.action_space.n

# network hyperparameters & initialise agent
alp = 0.0005
gam = 0.98
agent = Agent(state_space_size, action_space_size, alpha=alp, gamma=gam)

no_eps = 1000
marking = []
means = []
printing_rate = 50
marking_rate = 20
total_reward = 0.0

for ep in range(no_eps):
    epsilon = agent.select_epsilon(ep)
    state = env.reset()

    while True:
        action = agent.policy(state, epsilon)
        successor, reward, terminal, _ = env.step(action)

        terminal_mask = 0.0 if terminal else 1.0
        transition = (state, action, reward/100.0, successor, terminal_mask)
        agent.add_experience(transition)

        state = successor
        total_reward += reward

        if terminal:
            break

    agent.train_step(ep)

    marking.append(total_reward)
    if ep % marking_rate == 0 and ep != 0:
        std = np.array(marking).std()
        mean = np.array(marking).mean()
        means.append(mean)
        # print("marking, episode: {}, total reward: {:.1f}, mean score: {:.2f}, std score: {:.2f}"
        #       .format(ep, total_reward, mean, std))
        print("n_episode : {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
            ep, total_reward / marking_rate, agent.replay_buffer.size(), epsilon * 100))
        marking = []
        total_reward = 0.0

    # if ep % printing_rate == 0 and ep != 0:
        # print("episode: {}, total reward: {:.1f}, epsilon: {:.2f}".format(ep, total_reward, epsilon))

env.close()
enlapsed = time.time() - start
print("Time elapsed: ", enlapsed)

# Plot the mean reward array showing the progression of the agents success
plt.plot(means, color='green')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training progression [time = {:.2f}]'.format(enlapsed))
plt.savefig('training_progression_{}.png'.format(datetime.datetime.now()))
plt.show()
