import torch
import numpy as np
import pickle
import torch.nn as nn
import random
import math
from collections import deque


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
    def __init__(self, shape, device, pretrained=False, path=None, batch=32):
        self.epsilon = 0.02
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.error_limit = 1.0
        self.batch_size = batch
        self.state_shape = shape
        self.device = device
        self.pretrained = pretrained

        if self.pretrained:
            with open(path + "buffer.pkl", "rb") as f:
                self.tree = pickle.load(f)
            print("Loaded memory tree from path = {}".format(path + "buffer.pkl"))
        else:
            self.tree = BinarySumTree()
            print("Generated new memory tree")

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
                 buffer_capacity, batch_size, update_target, pretrained, path=None):

        self.state_shape = state_shape
        self.action_n = action_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_ceil
        self.epsilon_ceil = epsilon_ceil
        self.epsilon_floor = epsilon_floor
        self.epsilon_decay = epsilon_decay
        self.update_target = update_target
        self.pretrained = pretrained
        self.memory_capacity = buffer_capacity
        self.batch_size = batch_size

        self.timestep = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = nn.SmoothL1Loss().to(self.device)

        self.policy_network = Network(state_shape, action_n).to(self.device)
        self.target_network = Network(state_shape, action_n).to(self.device)

        if self.pretrained:
            self.policy_network.load_state_dict(torch.load(path + "policy_network.pt", map_location=torch.device(self.device)))
            self.policy_network.load_state_dict(torch.load(path + "target_network.pt", map_location=torch.device(self.device)))
            print("Loaded policy network from path = {}".format(path + "policy_network.pt"))
            print("Loaded target network from path = {}".format(path + "target_network.pt"))
        else:
            print("Generated randomly initiated new networks")

        self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=alpha)
        self.memory = PrioritisedMemory(self.state_shape, self.device, self.pretrained, path, self.batch_size)

    def step(self, state):
        self.timestep += 1

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

        if self.timestep % self.update_target == 0:
            print("updating target")
            self.target_update()
