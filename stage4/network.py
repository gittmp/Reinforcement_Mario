import torch
import numpy as np
import pickle
import torch.nn as nn
import random
import math


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
        self.maxlen = maxlen
        self.buffer = np.zeros(maxlen, dtype=object)
        self.pointer = 0
        self.full = False

        # generate tree with all node values 0 (maxlen nodes, each with 2 children, minus root)
        self.tree = torch.zeros(2 * maxlen - 1)

    def add(self, priority, data):
        # update data buffer
        self.buffer[self.pointer] = data

        # update tree leaves L -> R
        # fill leaves L -> R
        index = self.pointer + self.maxlen - 1
        self.update(index, priority)
        self.pointer += 1

        # if buffer is full, overwrite L -> R
        if self.pointer >= self.maxlen:
            self.full = True
            self.pointer = 0

    def update(self, i, p):
        # determine difference between old and new priority, then update
        diff = p - self.tree[i]
        self.tree[i] = p

        # backpropagate updates
        while i != 0:
            # need to update priority scores of internal nodes above updated leaf (as these sums depend on it)
            i = (i - 1) // 2
            # ISSUE with shape:
            # i = 50485; p = tensor([0.1148], grad_fn=<PowBackward0>); tree[i] = 2.0; diff = tensor([-0.8852], grad_fn=<SubBackward0>);
            self.tree[i] += diff

    def get_leaf(self, val):
        # output: leaf index, priority value, corresponding transition
        node = 0
        left = 1
        right = 2

        while left < len(self.tree):
            # search down the tree to find the highest priority node
            if val <= self.tree[left]:
                node = left
            else:
                val -= self.tree[left]
                node = right

            left = 2 * node + 1
            right = node + 1

        return node, self.tree[node], self.buffer[node - self.maxlen + 1]

    def total_priority(self):
        # return root node, i.e. sum of priorities
        return self.tree[0]


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
        if self.tree.full:
            return self.tree.maxlen
        else:
            return self.tree.pointer

    def push(self, experience):
        # retrieve the max priority
        maximum = torch.max(self.tree.tree[-self.tree.maxlen:])

        if maximum == 0:
            maximum = self.error_limit

        self.tree.add(maximum, experience)

    def sample(self):
        batch = []
        indices = torch.empty(self.batch_size, )
        weights = torch.empty(self.batch_size, 1)

        # increment beta each time we sample, annealing towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        # calculate maximum weight
        min_priority = torch.min(self.tree.tree[-self.tree.maxlen:]) / self.tree.total_priority()
        max_weight = pow(min_priority * self.batch_size, -self.beta)

        # divide range into sections
        section = math.floor(self.tree.total_priority() / self.batch_size)

        # uniformly sample transitions from each section
        for i in range(self.batch_size):
            low = section * i
            high = section * (i + 1)
            index, priority, transition = self.tree.get_leaf(np.random.uniform(low, high))

            p_j = priority / self.tree.total_priority()
            weights[i, 0] = pow(p_j * self.batch_size, -self.beta) / max_weight
            indices[i] = index
            batch.append([transition])

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

        return indices, batch, weights

    def update(self, indices, errors):
        for index, error in zip(indices, errors):
            error = min(float(error + self.epsilon), self.error_limit)
            priority = pow(error, self.alpha)
            self.tree.update(int(index), priority)


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
        self.memory.push(exp)

        if self.memory.size() < self.batch_size * 100:
            return

        # states, actions, rewards, successors, terminals = self.memory.sample()
        indices, batch, weights = self.memory.sample()
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        successors = batch['successors']
        terminals = batch['terminals']

        self.optimiser.zero_grad()

        q_vals = self.policy_network(states).gather(1, actions.long())
        targets = rewards + torch.mul((self.gamma * self.target_network(successors).max(1).values.unsqueeze(1)), 1 - terminals)

        abs_errors = torch.abs(targets - q_vals)
        self.memory.update(indices, abs_errors)

        loss = (weights * self.loss(q_vals, targets)).mean()
        loss.backward()
        self.optimiser.step()

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_floor)

        if self.timestep % self.update_target == 0:
            self.target_update()
