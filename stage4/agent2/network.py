import torch
import numpy as np
import pickle
import torch.nn as nn
import random
import collections


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


class Memory:
    def __init__(self, state_shape, buffer_capacity, batch_size, pretrained, device, source):

        self.batch_size = batch_size
        self.pretrained = pretrained
        self.state_shape = state_shape
        self.device = device
        self.source = source

        if self.pretrained:
            with open(self.source["path"] + "buffer.pkl", "rb") as f:
                self.buffer = pickle.load(f)

            self.buffer_capacity = self.buffer.maxlen

            with open(self.source["path"] + f'log4-{self.source["eps"]}.out', 'a') as f:
                f.write("Loaded buffer from path = {}".format(self.source["path"] + "buffer.pkl"))
        else:
            self.buffer = collections.deque(maxlen=buffer_capacity)
            self.buffer_capacity = buffer_capacity

            with open(self.source["path"] + f'log4-{self.source["eps"]}.out', 'a') as f:
                f.write("Generated new buffer")

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        state_batch = torch.zeros(self.batch_size, *self.state_shape)
        action_batch = torch.zeros(self.batch_size, 1)
        reward_batch = torch.zeros(self.batch_size, 1)
        successor_batch = torch.zeros(self.batch_size, *self.state_shape)
        terminal_batch = torch.zeros(self.batch_size, 1)

        for i in range(self.batch_size):
            s, a, r, succ, term = batch[i]
            state_batch[i] = s
            action_batch[i] = a
            reward_batch[i] = r
            successor_batch[i] = succ
            terminal_batch[i] = term

        return state_batch.to(self.device), \
               action_batch.to(self.device), \
               reward_batch.to(self.device), \
               successor_batch.to(self.device), \
               terminal_batch.to(self.device)


class Agent:
    def __init__(self, state_shape, action_n,
                 alpha, gamma, epsilon_ceil, epsilon_floor, epsilon_decay,
                 buffer_capacity, batch_size, update_target,
                 pretrained, source):

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
        self.source = source

        self.timestep = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = nn.SmoothL1Loss().to(self.device)

        self.policy_network = Network(state_shape, action_n).to(self.device)
        self.target_network = Network(state_shape, action_n).to(self.device)

        with open(self.source["path"] + f'log4-{self.source["eps"]}.out', 'a') as f:
            if self.pretrained:
                self.policy_network.load_state_dict(torch.load(self.source["path"] + "policy_network.pt", map_location=torch.device(self.device)))
                self.policy_network.load_state_dict(torch.load(self.source["path"] + "target_network.pt", map_location=torch.device(self.device)))

                f.write("Loaded policy network from path = {}".format(self.source["path"] + "policy_network.pt"))
                f.write("Loaded target network from path = {}".format(self.source["path"] + "target_network.pt"))
            else:
                f.write("Generated randomly initiated new networks")

        self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=alpha)

        self.memory = Memory(state_shape, buffer_capacity, batch_size, pretrained, self.device, self.source)

    def step(self, state):
        self.timestep += 1

        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.action_n)]])
        else:
            nn_out = self.policy_network(state.to(self.device))
            return torch.argmax(nn_out).unsqueeze(0).unsqueeze(0).cpu()

    def target_update(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        states, actions, rewards, successors, terminals = self.memory.sample()

        self.optimiser.zero_grad()

        q_vals = self.policy_network(states).gather(1, actions.long())
        targets = rewards + torch.mul((self.gamma * self.target_network(successors).max(1).values.unsqueeze(1)), 1 - terminals)

        loss = self.loss(q_vals, targets)
        loss.backward()
        self.optimiser.step()

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_floor)

        if self.timestep % self.update_target == 0:
            self.target_update()
