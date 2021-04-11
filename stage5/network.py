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
    b = b and os.path.isfile(path + "episode_rewards.pkl")
    # b = b and os.path.isfile(path + "buffer.pkl")

    return b


class Network0(nn.Module):
    def __init__(self, in_features, n_actions):
        super(Network0, self).__init__()

        # VERSION: architecture from 'Playing Atari with Deep Reinforcement Learning'
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


class Network1(nn.Module):
    def __init__(self, in_features, n_actions):
        super(Network1, self).__init__()
        self.in_features = in_features

        # VERSION: architecture from 'DDDQN (Double Dueling Deep Q Learning with Prioritized Experience Replay)'

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features[0], out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_out = self.conv3(self.conv2(self.conv1(torch.zeros(1, *in_features))))
        conv_out_size = int(np.prod(conv_out.size()))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    # forward pass combining conv set and lin set
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)

        return x


class BasicMemory:
    def __init__(self, state_shape, buffer_capacity, batch_size, pretrained, device, path, eps):

        self.batch_size = batch_size
        self.pretrained = pretrained
        self.state_shape = state_shape
        self.device = device
        self.path = path
        self.n_eps = eps

        # if self.pretrained:
        #     with open(path + "buffer.pkl", "rb") as f:
        #         self.buffer = pickle.load(f)
        #
        #     self.buffer_capacity = self.buffer.maxlen
        #
        #     with open(self.path + 'log.out', 'a') as f:
        #         f.write("Loaded buffer from path = {}".format(self.path + "buffer.pkl"))
        # else:

        self.buffer = deque(maxlen=buffer_capacity)
        self.buffer_capacity = buffer_capacity

        with open(self.path + 'log.out', 'a') as f:
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

    def size(self):
        return len(self.buffer)


class TreeStruct(object):
    def __init__(self, maxlen=100000):
        # specify the max number of leaves holding priorities, and buffer holding experiences
        self.priority_limit = 1.0
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.tree = deque(maxlen=maxlen)

    def add(self, data):
        maximum = self.max_priority()

        self.buffer.append(data)
        self.tree.append(maximum)

    def update(self, i, e):
        self.tree[i] = e

    def get_leaves(self, n):
        priorities = []
        batch = []
        indices = torch.empty(n, )

        sum_priorities = sum(self.tree)
        priority_distribution = [p / sum_priorities for p in self.tree]

        inds = np.random.choice(self.size(), size=n, p=priority_distribution)

        for i in range(n):
            index = inds[i]
            indices[i] = index

            transition = self.buffer[index]
            batch.append([transition])

            e = self.tree[index]
            priorities.append(e)

        return indices, priorities, batch

    def max_priority(self):
        if len(self.tree) > 0:
            m = max(self.tree)

            if m == 0:
                m = self.priority_limit
        else:
            m = self.priority_limit

        return m

    def size(self):
        return len(self.tree)

    def total_priority(self):
        return sum(self.tree)


class PrioritisedMemory:
    def __init__(self, shape, device, eps, batch=64, pretrained=False, path=None):
        self.epsilon = 0.02
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.priority_limit = 1.0
        self.batch_size = batch
        self.state_shape = shape
        self.n_eps = eps
        self.device = device
        self.pretrained = pretrained
        self.path = path

        # if self.pretrained:
        #     with open(self.path + "buffer.pkl", "rb") as f:
        #         self.tree = pickle.load(f)
        #
        #     with open(self.path + 'log.out', 'a') as f:
        #         f.write("Loaded memory tree from path = {} \n".format(self.path + "buffer.pkl"))
        # else:

        self.tree = TreeStruct()

        with open(self.path + 'log.out', 'a') as f:
            f.write("Generated new memory tree \n")

    def size(self):
        return len(self.tree.tree)

    def push(self, experience):
        self.tree.add(experience)

    def sample(self):
        # increment beta each time we sample, annealing towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        # uniformly sample transitions from each priority section
        indices, p_i_array, batch = self.tree.get_leaves(self.batch_size)

        weights = torch.empty(self.batch_size, 1).to(self.device)
        N = self.tree.size()
        total_priority = sum(p_i_array)

        for i in range(self.batch_size):
            # w = (1 / N*p_i)^B -> normalise to [0, 1]
            P_i = p_i_array[i] / total_priority
            w_i = pow((N * P_i), -self.beta)
            weights[i, 0] = w_i

        total_weight = sum([w[0] for w in weights])

        for i in range(self.batch_size):
            weights[i, 0] = weights[i, 0] / total_weight

        # convert batch to torch
        state_batch = torch.zeros(self.batch_size, *self.state_shape).to(self.device)
        action_batch = torch.zeros(self.batch_size, 1).to(self.device)
        reward_batch = torch.zeros(self.batch_size, 1).to(self.device)
        successor_batch = torch.zeros(self.batch_size, *self.state_shape).to(self.device)
        terminal_batch = torch.zeros(self.batch_size, 1).to(self.device)

        for i in range(self.batch_size):
            state_batch[i], action_batch[i], reward_batch[i],  successor_batch[i], terminal_batch[i] = batch[i][0]

        batch = {
            'states': state_batch,
            'actions': action_batch,
            'rewards': reward_batch,
            'successors': successor_batch,
            'terminals': terminal_batch
        }

        return indices.to(self.device), batch, weights

    def update(self, indices, errors):
        for index, error in zip(indices, errors):
            p = pow(float(error) + self.epsilon, self.alpha)
            self.tree.update(int(index), p)


class Agent:
    def __init__(self, state_shape, action_n,
                 alpha, gamma, epsilon_ceil, epsilon_floor, epsilon_decay,
                 buffer_capacity, batch_size, update_target, path, episodes,
                 pretrained=False, plot=False, training=False,
                 network=1, memory=1):

        self.n_eps = episodes
        self.path = path
        self.version = memory
        self.timestep_array = []
        self.intrinsic_rewards = []

        self.state_shape = state_shape
        self.action_n = action_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_ceil
        self.epsilon_ceil = epsilon_ceil
        self.epsilon_floor = epsilon_floor
        self.epsilon_decay = epsilon_decay
        self.update_target = update_target
        self.pretrained = pretrained and check_files(self.path)
        self.memory_capacity = buffer_capacity
        self.batch_size = batch_size

        self.plot = plot
        self.training = training

        self.train_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.version == 1:
            self.loss = nn.SmoothL1Loss(reduction='none').to(self.device)
        else:  # self.version == 0
            self.loss = nn.SmoothL1Loss().to(self.device)

        if network == 0:
            self.policy_network = Network0(state_shape, action_n).to(self.device)
            self.target_network = Network0(state_shape, action_n).to(self.device)
        else:  # network == 1
            self.policy_network = Network1(state_shape, action_n).to(self.device)
            self.target_network = Network1(state_shape, action_n).to(self.device)

        if self.pretrained:
            self.policy_network.load_state_dict(torch.load(self.path + "policy_network.pt", map_location=torch.device(self.device)))
            self.policy_network.load_state_dict(torch.load(self.path + "target_network.pt", map_location=torch.device(self.device)))

            with open(self.path + 'log.out', 'a') as f:
                f.write("\nLoaded policy network from path = {} \n".format(self.path + "policy_network.pt"))
                f.write("Loaded target network from path = {} \n".format(self.path + "target_network.pt"))

            with open(self.path + "episode_rewards.pkl", "rb") as f:
                self.extrinsic_rewards = pickle.load(f)

            with open(self.path + 'log.out', 'a') as f:
                f.write("Loaded rewards over {} episodes from path = {} \n".format(len(self.extrinsic_rewards), self.path))
        else:
            self.extrinsic_rewards = []

            with open(self.path + 'log.out', 'a') as f:
                f.write("\nGenerated randomly initiated new networks \n")

        self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=alpha)

        if self.version == 1:
            self.memory = PrioritisedMemory(self.state_shape, self.device, self.n_eps, self.batch_size, self.pretrained, self.path)
        else:  # self.version == 0
            self.memory = BasicMemory(state_shape, buffer_capacity, batch_size, pretrained, self.device, self.path, self.n_eps)

    def step(self, state):
        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.action_n)]])
        else:
            nn_out = self.policy_network(state.to(self.device))
            return torch.argmax(nn_out).unsqueeze(0).unsqueeze(0).cpu()

    def target_update(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def train(self, exp, ep):
        self.train_step += 1
        self.memory.push(exp)

        if self.memory.size() < self.batch_size * 100:
            print("\rCan't train on episode {} - MEMORY TOO SMALL: size = {}, target = {}".format(ep, self.memory.size(), self.batch_size * 100), end='', flush=True)
            return

        print("\rTraining on step {} in episode {}".format(self.train_step, ep), end='', flush=True)

        # sample from memory
        if self.version == 1:
            indices, batch, weights = self.memory.sample()
            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards']
            successors = batch['successors']
            terminals = batch['terminals']
        else:  # self.version == 0
            states, actions, rewards, successors, terminals = self.memory.sample()

        self.optimiser.zero_grad()

        # TD target = reward + discount factor * successor q-value
        targets = (rewards + torch.mul((self.gamma * self.target_network(successors).max(1).values.unsqueeze(1)), 1 - terminals)).to(self.device)

        # TD error = TD target - prev. q-value
        q_vals = self.policy_network(states).gather(1, actions.long()).to(self.device)

        if self.version == 1:
            td_errors = torch.abs(targets - q_vals)
            self.memory.update(indices, td_errors)
            loss = (weights * self.loss(q_vals, targets)).mean()
        else:  # self.version == 0
            loss = self.loss(q_vals, targets)

        loss.backward()
        self.optimiser.step()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_floor)

        if self.train_step % self.update_target == 0:
            self.target_update()

    def save(self):
        with open(self.path + "episode_rewards.pkl", "wb") as f:
            pickle.dump(self.extrinsic_rewards, f)

        # with open(self.path + "buffer.pkl", "wb") as f:
        #     if self.version == 1:
        #         pickle.dump(self.memory.tree, f)
        #     else:  # self.version == 0
        #         pickle.dump(self.memory.buffer, f)

        torch.save(self.policy_network.state_dict(), self.path + "policy_network.pt")
        torch.save(self.target_network.state_dict(), self.path + "target_network.pt")

    def run(self, env, eps):

        self.timestep_array = []
        self.intrinsic_rewards = []

        for ep in tqdm(range(eps)):

            state = env.reset()
            state = torch.Tensor([state])
            timestep = 0
            total_reward = 0

            while True:
                timestep += 1

                if self.plot:
                    env.render()

                    # if timestep % 10 == 0:
                    #     render_state(state)

                action = self.step(state)

                successor, reward, terminal, info = env.step(int(action[0]))
                successor = torch.Tensor([successor])
                total_reward += reward

                if self.training:
                    experience = (
                        state.float(),
                        action.float(),
                        torch.Tensor([reward]).unsqueeze(0).float(),
                        successor.float(),
                        torch.Tensor([int(terminal)]).unsqueeze(0).float()
                    )

                    self.train(experience, ep)

                state = successor

                if terminal:
                    self.timestep_array.append(timestep)
                    self.intrinsic_rewards.append(total_reward)
                    self.extrinsic_rewards.append(info['score'])
                    break

            if self.training:
                if ep % max(1, math.floor(eps / 4)) == 0:
                    with open(self.path + 'log.out', 'a') as f:
                        f.write("automatically saving params4 at episode {} \n".format(ep))

                    self.save()

        if self.training:
            with open(self.path + 'log.out', 'a') as f:
                f.write("\nSaving final parameters! \n")
            self.save()

    def print_stats(self, no_plot_points=25):

        with open(self.path + 'log.out', 'a') as f:
            f.write("\nTotal episodes trained over: {} \n".format(len(self.extrinsic_rewards)))

            sections = min(no_plot_points, self.n_eps)
            section_size = math.floor(len(self.extrinsic_rewards) / sections)

            # print table of extrinsic rewards (game scores) over session
            f.write("\n\nAverage extrinsic rewards over past {} episodes: \n".format(len(self.extrinsic_rewards)))
            f.write("EPISODE RANGE                AV. EXTRINSIC REWARD \n")

            for i in range(sections):
                low = i * section_size
                high = (i + 1) * section_size

                if i == sections - 1:
                    av = sum(self.extrinsic_rewards[low:]) / (len(self.extrinsic_rewards) - low)
                    f.write("[{}, {}) {} {} \n".format(low, len(self.extrinsic_rewards), " " * (25 - 2 - len(str(low)) - len(str(len(self.extrinsic_rewards)))), av))
                else:
                    av = sum(self.extrinsic_rewards[low:high]) / (high - low)
                    f.write("[{}, {}) {} {} \n".format(low, high, " " * (25 - 2 - len(str(low)) - len(str(high))), av))

            # print table of intrinsic rewards (manufactured reward signal) over session
            f.write("\n\nAverage intrinsic rewards over past {} episodes: \n".format(len(self.intrinsic_rewards)))
            f.write("EPISODE RANGE                AV. INTRINSIC REWARD \n")

            for i in range(sections):
                low = i * section_size
                high = (i + 1) * section_size

                if i == sections - 1:
                    av = sum(self.intrinsic_rewards[low:]) / (len(self.intrinsic_rewards) - low)
                    f.write("[{}, {}) {} {} \n".format(low, len(self.intrinsic_rewards), " " * (25 - 2 - len(str(low)) - len(str(len(self.intrinsic_rewards)))), av))
                else:
                    av = sum(self.intrinsic_rewards[low:high]) / (high - low)
                    f.write("[{}, {}) {} {} \n".format(low, high, " " * (25 - 2 - len(str(low)) - len(str(high))), av))

            # print table of episode lengths over session
            f.write("\n\nAverage episode length (no. timesteps) over past {} episodes: \n".format(len(self.timestep_array)))
            f.write("EPISODE RANGE                AV. LENGTH \n")

            for i in range(sections):
                low = i * section_size
                high = (i + 1) * section_size

                if i == sections - 1:
                    av = sum(self.timestep_array[low:]) / (len(self.timestep_array) - low)
                    f.write("[{}, {}) {} {} \n".format(low, len(self.timestep_array), " " * (25 - 2 - len(str(low)) - len(str(len(self.timestep_array)))), av))
                else:
                    av = sum(self.timestep_array[low:high]) / (high - low)
                    f.write("[{}, {}) {} {} \n".format(low, high, " " * (25 - 2 - len(str(low)) - len(str(high))), av))
