import torch
import numpy as np
import pickle
import torch.nn as nn
import random
import math
import os
from collections import deque
from tqdm import tqdm
# from environment import plot_durations


def check_files(path):
    b = os.path.isfile(path + "policy_network.pt")
    b = b and os.path.isfile(path + "target_network.pt")
    b = b and os.path.isfile(path + "buffer.pkl")
    b = b and os.path.isfile(path + "episode_rewards.pkl")

    return b


class Network(nn.Module):
    def __init__(self, in_features, n_actions):
        super(Network, self).__init__()

        # VERSION: testing architecture from 'Playing Atari with Deep Reinforcement Learning'
        self.conv = nn.Sequential(
            nn.Conv2d(in_features[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )

        conv_out = self.conv(torch.zeros(1, *in_features))
        conv_out_size = int(np.prod(conv_out.size()))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    # forward pass combining conv set and lin set
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


class BinarySumTree(object):
    def __init__(self, maxlen=100000):
        # specify the max number of leaves holding priorities, and buffer holding experiences
        self.error_limit = 1.0
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.tree = deque(maxlen=maxlen)

    def add(self, error, data):
        self.buffer.append(data)
        self.tree.append(error)

    def update(self, i, e):
        self.tree[i] = e

    def get_leaf(self, l, h):
        # retrieve random priority error in range low-high
        val = np.random.uniform(l, h)
        err = 0
        R = len(self.tree)

        for i in range(R):
            if err < self.tree[i] < val:
                err = self.tree[i]
                node = i

        trans = self.buffer[node]

        return node, float(err), trans

    def max_priority(self):
        if len(self.tree) > 0:
            m = max(self.tree)

            if m == 0:
                m = self.error_limit
        else:
            m = self.error_limit

        return m

    def size(self):
        return len(self.tree)

    def total_priority(self):
        return sum(self.tree)


class PrioritisedMemory:
    def __init__(self, shape, device, eps, batch=32, pretrained=False, path=None):
        self.epsilon = 0.02
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.error_limit = 1.0
        self.batch_size = batch
        self.state_shape = shape
        self.n_eps = eps
        self.device = device
        self.pretrained = pretrained
        self.path = path

        if self.pretrained:
            with open(self.path + "buffer.pkl", "rb") as f:
                self.tree = pickle.load(f)

            with open(self.path + f'log4-{self.n_eps}.out', 'a') as f:
                f.write("Loaded memory tree from path = {}".format(self.path + "buffer.pkl"))
        else:
            self.tree = BinarySumTree()

            with open(self.path + f'log4-{self.n_eps}.out', 'a') as f:
                f.write("Generated new memory tree")

    def size(self):
        return len(self.tree.tree)

    def push(self, experience):
        # retrieve the max priority
        maximum = self.tree.max_priority()
        self.tree.add(maximum, experience)

    def sample(self):
        batch = []
        indices = torch.empty(self.batch_size, )
        weights = torch.empty(self.batch_size, 1)

        # increment beta each time we sample, annealing towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        # uniformly sample transitions from each priority section
        priorities = []
        section_length = self.tree.max_priority() / self.batch_size
        N = self.tree.size()
        low = 0

        for i in range(self.batch_size):
            high = (i + 1) * section_length
            index, error, transition = self.tree.get_leaf(low, high)

            p_i = float(error) + self.epsilon
            p_i_a = pow(p_i, self.alpha)

            priorities.append(p_i_a)
            indices[i] = index
            batch.append([transition])

        total_priority = sum(priorities)
        # min_p = min(priorities) / total_priority
        # max_w = pow((1/N) * (1/min_p), self.beta)

        for i in range(self.batch_size):
            # w = (1 / N*p_i)^B -> normalise to [0, 1]
            P_i = priorities[i] / total_priority
            w_i = pow((1/N) * (1/P_i), self.beta)
            # w_i = w_i / max_w
            weights[i, 0] = w_i

        # convert batch to torch
        state_batch = torch.zeros(self.batch_size, *self.state_shape)
        action_batch = torch.zeros(self.batch_size, 1)
        reward_batch = torch.zeros(self.batch_size, 1)
        successor_batch = torch.zeros(self.batch_size, *self.state_shape)
        terminal_batch = torch.zeros(self.batch_size, 1)

        for i in range(len(batch)):
            item = batch[i]

            state_batch[i] = item[0][0]
            action_batch[i] = item[0][1]
            reward_batch[i] = item[0][2]
            successor_batch[i] = item[0][3]
            terminal_batch[i] = item[0][4]

        batch = {
            'states': state_batch.to(self.device),
            'actions': action_batch.to(self.device),
            'rewards': reward_batch.to(self.device),
            'successors': successor_batch.to(self.device),
            'terminals': terminal_batch.to(self.device)
        }

        return indices.to(self.device), batch, weights.to(self.device)

    def update(self, indices, errors):
        for index, error in zip(indices, errors):
            self.tree.update(int(index), error)


class Agent:
    def __init__(self, state_shape, action_n,
                 alpha, gamma, epsilon_ceil, epsilon_floor, epsilon_decay,
                 buffer_capacity, batch_size, update_target, eps,
                 pretrained=False, path=None, plot=False, training=False):

        self.state_shape = state_shape
        self.action_n = action_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_ceil
        self.epsilon_ceil = epsilon_ceil
        self.epsilon_floor = epsilon_floor
        self.epsilon_decay = epsilon_decay
        self.update_target = update_target
        self.pretrained = pretrained and check_files(path)
        self.memory_capacity = buffer_capacity
        self.batch_size = batch_size

        self.n_eps = eps
        self.path = path
        self.plot = plot
        self.training = training

        self.train_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = nn.SmoothL1Loss().to(self.device)

        self.policy_network = Network(state_shape, action_n).to(self.device)
        self.target_network = Network(state_shape, action_n).to(self.device)

        if self.pretrained:
            self.policy_network.load_state_dict(torch.load(self.path + "policy_network.pt", map_location=torch.device(self.device)))
            self.policy_network.load_state_dict(torch.load(self.path + "target_network.pt", map_location=torch.device(self.device)))

            with open(self.path + f'log4-{self.n_eps}.out', 'a') as f:
                f.write("Loaded policy network from path = {}".format(self.path + "policy_network.pt"))
                f.write("Loaded target network from path = {}".format(self.path + "target_network.pt"))

            with open(self.path + "episode_rewards.pkl", "rb") as f:
                self.episode_rewards = pickle.load(f)

            with open(self.path + f'log4-{self.n_eps}.out', 'a') as f:
                f.write("Loaded rewards over {} episodes from path = {}".format(len(self.episode_rewards), path))
        else:
            self.episode_rewards = []

            with open(self.path + f'log4-{self.n_eps}.out', 'a') as f:
                f.write("Generated randomly initiated new networks")

        self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=alpha)
        self.memory = PrioritisedMemory(self.state_shape, self.device, self.n_eps, self.batch_size, self.pretrained, path)

    def step(self, state):
        self.train_step += 1

        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.action_n)]])
        else:
            nn_out = self.policy_network(state.to(self.device))
            return torch.argmax(nn_out).unsqueeze(0).unsqueeze(0).cpu()

    def target_update(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def train(self, exp):

        if self.memory.size() < self.batch_size * 100:
            return

        # sample from memory
        indices, batch, weights = self.memory.sample()
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        successors = batch['successors']
        terminals = batch['terminals']

        self.optimiser.zero_grad()

        # TD target = reward + discount factor * successor q-value
        targets = (rewards + torch.mul((self.gamma * self.target_network(successors).max(1).values.unsqueeze(1)), 1 - terminals)).to(self.device)

        # TD error = TD target - prev. q-value
        q_vals = self.policy_network(states).gather(1, actions.long()).to(self.device)
        abs_errors = torch.abs(targets - q_vals)
        self.memory.update(indices, abs_errors)

        td_loss = self.loss(q_vals, targets)

        loss = (weights * td_loss).mean()
        loss.backward()
        self.optimiser.step()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_floor)

        # append experience to memory AFTER update so that indices in deque are the same
        self.memory.push(exp)

        if self.train_step % self.update_target == 0:
            self.target_update()

    def save(self):
        with open(self.path + "episode_rewards.pkl", "wb") as f:
            pickle.dump(self.episode_rewards, f)

        with open(self.path + "buffer.pkl", "wb") as f:
            pickle.dump(self.memory.tree, f)

        torch.save(self.policy_network.state_dict(), self.path + "policy_network.pt")
        torch.save(self.target_network.state_dict(), self.path + "target_network.pt")

    def run(self, env, eps):

        for ep in tqdm(range(eps)):
            state = env.reset()
            state = torch.Tensor([state])
            # total_reward = 0
            timestep = 0

            while True:
                timestep += 1

                if self.plot:
                    env.render()

                    # if timestep % 10 == 0:
                    #     render_state(state)

                action = self.step(state)

                successor, reward, terminal, info = env.step(int(action[0]))
                successor = torch.Tensor([successor])

                if self.training:
                    experience = (
                        state.float(),
                        action.float(),
                        torch.Tensor([reward]).unsqueeze(0).float(),
                        successor.float(),
                        torch.Tensor([int(terminal)]).unsqueeze(0).float()
                    )

                    self.train(experience)

                state = successor

                if terminal:
                    break

            if self.training:
                self.episode_rewards.append(info['score'])

                with open(self.path + f'log4-{self.n_eps}.out', 'a') as f:
                    f.write("\nGame score after termination = {}".format(info['score']))

                if ep % max(1, math.floor(eps / 4)) == 0:
                    with open(self.path + f'log4-{self.n_eps}.out', 'a') as f:
                        f.write("automatically saving params4 at episode {}".format(ep))

                    self.save()

        if self.training:
            with open(self.path + f'log4-{self.n_eps}.out', 'a') as f:
                f.write("\nSaving final parameters!")
            self.save()

    def print_stats(self, eps):
        with open(self.path + f'log4-{eps}.out', 'a') as f:
            f.write("Total episodes trained over: {}".format(len(self.episode_rewards)))

            sections = 10
            section_size = math.floor(len(self.episode_rewards) / sections)
            low = 0

            f.write("\nAverage environment rewards over past {} episodes:".format(len(self.episode_rewards)))
            f.write("EPISODE RANGE                AV. REWARD")

            for i in range(sections):
                high = (i + 1) * section_size

                if i == sections - 1:
                    av = sum(self.episode_rewards[low:]) / (len(self.episode_rewards) - low)
                    f.write("[{}, {}) {} {}".format(low, len(self.episode_rewards), " " * (25 - 2 - len(str(low)) - len(str(len(self.episode_rewards)))), av))
                else:
                    av = sum(self.episode_rewards[low:high]) / (high - low)
                    f.write("[{}, {}) {} {}".format(low, high, " " * (25 - 2 - len(str(low)) - len(str(high))), av))

                low = high

