import torch
import numpy as np
import pickle
import torch.nn as nn
import random
import math
import os
from collections import deque
from tqdm import tqdm
import time
# from environment import render_state


def check_files(path):
    b = os.path.isfile(path + "policy_network.pt")
    b = b and os.path.isfile(path + "target_network.pt")
    b = b and os.path.isfile(path + "extrinsic_rewards.pkl")

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

        # VERSION: adapted architecture
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
    def __init__(self, state_shape, maxlen, batch_size, pretrained, device, path, eps):
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.state_shape = state_shape
        self.device = device
        self.path = path
        self.n_eps = eps

        self.buffer = deque(maxlen=maxlen)
        self.maxlen = maxlen

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

        batch = {
            'states': state_batch.to(self.device),
            'actions': action_batch.to(self.device),
            'rewards': reward_batch.to(self.device),
            'successors': successor_batch.to(self.device),
            'terminals': terminal_batch.to(self.device)
        }

        return batch

    def size(self):
        return len(self.buffer)


class SumTreeStructure:
    def __init__(self, maxlen, device):
        self.device = device
        self.maxlen = maxlen
        self.tree = torch.zeros(2 * maxlen - 1).to(self.device)
        self.buffer = np.zeros(maxlen, dtype=object)
        self.length = 0
        self.pointer = 0

    # update value at internal nodes until new total held at root
    def propagate(self, ind, delta):
        parent = (ind - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self.propagate(parent, delta)

    # return an index of a sampled element, given a random input value in current section
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= float(self.tree[left]):
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - float(self.tree[left]))

    # get priority and sample
    def get(self, l, h):
        exp = 0
        count = 0
        while exp == 0 and count < 64:
            count += 1
            v = random.uniform(l, h)
            idx = self._retrieve(0, v)
            dataIdx = idx - self.maxlen + 1
            exp = self.buffer[dataIdx]

        if count == 64:
            print("CANT FIND EXPERIENCE TO SAMPLE")
            exit(1)

        return idx, float(self.tree[idx]), exp

    def total(self):
        return float(self.tree[0])

    def full(self):
        return self.length >= self.maxlen

    # a new experience with its associated priority
    def push(self, pri, exp):
        ind = self.pointer + self.maxlen - 1

        if type(exp) != tuple or len(exp) != 5:
            print(f"tree push - ERRONEOUS EXPERIENCE ADDED: \n{exp}")
            exit(1)

        self.buffer[self.pointer] = exp
        self.update(ind, pri)

        self.pointer += 1
        if self.pointer >= self.maxlen:
            self.pointer = 0

        if self.length < self.maxlen:
            self.length += 1

    # update priority associated with some index
    def update(self, ind, pri):
        delta = pri - self.tree[ind]
        self.tree[ind] = pri
        self.propagate(ind, delta)


class PrioritisedMemory:
    def __init__(self, maxlen, shape, dev):
        self.tree = SumTreeStructure(maxlen, dev)
        self.maxlen = maxlen
        self.state = shape
        self.device = dev

        self.epsilon = 0.01
        self.alpha = 0.8
        self.tau = 0.3
        self.tau_inc = 0.0005

    def push(self, error, sample):
        p = pow(float(torch.abs(error)) + self.epsilon, self.alpha)

        if type(sample) != tuple or len(sample) != 5:
            print(f"PER push - ERRONEOUS EXPERIENCE ADDED: \n{sample}")
            exit(1)

        self.tree.push(p, sample)

    def full(self):
        return self.tree.full()

    def sample(self, n):
        segment = self.tree.total() / n
        indices = torch.zeros(n).to(self.device)
        priorities = torch.zeros(n).to(self.device)

        self.tau = np.min([1.0, self.tau + self.tau_inc])

        # convert batch to torch
        state_batch = torch.zeros(n, *self.state).to(self.device)
        action_batch = torch.zeros(n, 1).to(self.device)
        reward_batch = torch.zeros(n, 1).to(self.device)
        successor_batch = torch.zeros(n, *self.state).to(self.device)
        terminal_batch = torch.zeros(n, 1).to(self.device)

        for i in range(n):
            low = segment * i
            high = segment * (i + 1)

            index, priority, experience = self.tree.get(low, high)
            priorities[i] = priority
            indices[i] = index

            if type(experience) != tuple or len(experience) != 5:
                print(f"sample - ERRONEOUS EXPERIENCE ADDED: \n{experience}")
                exit(1)

            state_batch[i], action_batch[i], reward_batch[i], successor_batch[i], terminal_batch[i] = experience

        sampling_probabilities = torch.div(priorities, self.tree.total())
        weights = torch.mul(sampling_probabilities, float(self.tree.length)).to(self.device)
        weights = torch.pow(weights, -self.tau)
        weights = torch.div(weights, weights.max())

        batch = {
            'states': state_batch,
            'actions': action_batch,
            'rewards': reward_batch,
            'successors': successor_batch,
            'terminals': terminal_batch
        }

        return batch, indices, weights

    def update(self, indices, errors):
        for i, e in zip(indices, errors):
            p = pow(float(torch.abs(e)) + self.epsilon, self.alpha)
            self.tree.update(int(i), p)


class Agent:
    def __init__(self, state_shape, action_n,
                 alpha, gamma, epsilon_ceil, epsilon_floor, epsilon_decay,
                 buffer_capacity, batch_size, update_target, path, episodes,
                 pretrained=False, plot=False, training=False,
                 network=1, memory=2, env_version=2):

        self.n_eps = episodes
        self.path = path
        self.mem_version = memory
        self.env_version = env_version
        self.training_times = []

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
        self.batch_size = batch_size

        self.plot = plot
        self.training = training

        self.train_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.mem_version == 2:
            self.loss = nn.SmoothL1Loss(reduction='none').to(self.device)
        else:  # self.version == 0 or 1
            self.loss = nn.SmoothL1Loss().to(self.device)

        if network == 0:
            self.policy_network = Network0(state_shape, action_n).to(self.device)
            self.target_network = Network0(state_shape, action_n).to(self.device)
        else:  # network == 1
            self.policy_network = Network1(state_shape, action_n).to(self.device)
            self.target_network = Network1(state_shape, action_n).to(self.device)

        if self.pretrained:
            self.policy_network.load_state_dict(torch.load(self.path + "policy_network.pt", map_location=torch.device(self.device)))
            self.target_network.load_state_dict(torch.load(self.path + "target_network.pt", map_location=torch.device(self.device)))

            with open(self.path + 'log.out', 'a') as f:
                f.write("\nLoaded policy network from path = {} \n".format(self.path + "policy_network.pt"))
                f.write("Loaded target network from path = {} \n".format(self.path + "target_network.pt"))

            with open(self.path + "extrinsic_rewards.pkl", "rb") as f:
                self.extrinsic_rewards = pickle.load(f)

            if env_version == 2:
                with open(self.path + "intrinsic_rewards.pkl", "rb") as f:
                    self.intrinsic_rewards = pickle.load(f)

            with open(self.path + "distance_array.pkl", "rb") as f:
                self.distance_array = pickle.load(f)

            with open(self.path + 'log.out', 'a') as f:
                f.write("Loaded rewards over {} episodes from path = {} \n".format(len(self.extrinsic_rewards), self.path))
        else:
            self.extrinsic_rewards = []
            self.distance_array = []

            if env_version == 2:
                self.intrinsic_rewards = []

        self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=alpha)

        if self.mem_version == 2:
            self.memory = PrioritisedMemory(100000, self.state_shape, self.device)
        elif self.mem_version == 1:
            self.memory = BasicMemory(self.state_shape, buffer_capacity, self.batch_size, self.pretrained, self.device, self.path, self.n_eps)

    def step(self, state):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_floor)

        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.action_n)]])
        else:
            nn_out = self.policy_network(state.to(self.device))
            return torch.argmax(nn_out).unsqueeze(0).unsqueeze(0).cpu()

    def target_update(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def train(self, exp):
        self.train_step += 1

        # sample from memory
        if self.mem_version == 2:
            S = torch.zeros(1, *self.state_shape).to(self.device)
            A = torch.zeros(1, 1).to(self.device)
            R = torch.zeros(1, 1).to(self.device)
            Succ = torch.zeros(1, *self.state_shape).to(self.device)
            T = torch.zeros(1, 1).to(self.device)
            S[0], A[0], R[0], Succ[0], T[0] = exp

            target = (R + torch.mul((self.gamma * self.target_network(Succ).max(1).values.unsqueeze(1)), 1 - T)).to(self.device)
            q_val = self.policy_network(S).gather(1, A.long()).to(self.device)
            td_error = torch.abs(target - q_val)
            self.memory.push(td_error, exp)

            if not self.memory.full():
                return

            batch, indices, weights = self.memory.sample(self.batch_size)

            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards']
            successors = batch['successors']
            terminals = batch['terminals']
        elif self.mem_version == 1:
            self.memory.push(exp)

            if self.memory.size() < self.batch_size * 100:
                return

            batch = self.memory.sample()
            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards']
            successors = batch['successors']
            terminals = batch['terminals']
        else:  # self.mem_version == 0
            states = torch.zeros(1, *self.state_shape).to(self.device)
            actions = torch.zeros(1, 1).to(self.device)
            rewards = torch.zeros(1, 1).to(self.device)
            successors = torch.zeros(1, *self.state_shape).to(self.device)
            terminals = torch.zeros(1, 1).to(self.device)

            states[0], actions[0], rewards[0], successors[0], terminals[0] = exp

        # TD target = reward + discount factor * successor q-value
        targets = (rewards + torch.mul((self.gamma * self.target_network(successors).max(1).values.unsqueeze(1)), 1 - terminals)).to(self.device)

        # TD error = TD target - prev. q-value
        q_vals = self.policy_network(states).gather(1, actions.long()).to(self.device)

        if self.mem_version == 2:
            td_errors = torch.abs(targets - q_vals)
            self.memory.update(indices, td_errors)

            loss = self.loss(q_vals, targets)
            loss = torch.mul(loss, weights).mean()
        else:  # self.version == 0 or 1
            loss = self.loss(q_vals, targets)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        if self.train_step % self.update_target == 0:
            self.target_update()

    def save(self):
        torch.save(self.policy_network.state_dict(), self.path + "policy_network.pt")
        torch.save(self.target_network.state_dict(), self.path + "target_network.pt")

        with open(self.path + "extrinsic_rewards.pkl", "wb") as f:
            pickle.dump(self.extrinsic_rewards, f)

        if self.env_version == 2:
            with open(self.path + "intrinsic_rewards.pkl", "wb") as f:
                pickle.dump(self.intrinsic_rewards, f)

        with open(self.path + "distance_array.pkl", "wb") as f:
            pickle.dump(self.distance_array, f)

    def run(self, env, eps):

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
                    start = time.time()

                    experience = (
                        state.float(),
                        action.float(),
                        torch.Tensor([reward]).unsqueeze(0).float(),
                        successor.float(),
                        torch.Tensor([int(terminal)]).unsqueeze(0).float()
                    )
                    self.train(experience)

                    self.training_times.append(time.time() - start)

                state = successor

                if terminal:
                    if self.env_version == 2:
                        self.intrinsic_rewards.append(total_reward)

                    self.extrinsic_rewards.append(info['score'])
                    self.distance_array.append(info['x_pos'])
                    break

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

            if self.env_version == 2:
                # print table of intrinsic rewards (manufactured reward signal) over session
                f.write("\n\nAverage intrinsic rewards over past {} episodes: \n".format(len(self.intrinsic_rewards)))
                f.write("EPISODE RANGE                AV. INTRINSIC REWARD \n")
                section_size = math.floor(len(self.intrinsic_rewards) / sections)

                for i in range(sections):
                    low = i * section_size
                    high = (i + 1) * section_size

                    if i == sections - 1:
                        av = sum(self.intrinsic_rewards[low:]) / (len(self.intrinsic_rewards) - low)
                        f.write("[{}, {}) {} {} \n".format(low, len(self.intrinsic_rewards), " " * (25 - 2 - len(str(low)) - len(str(len(self.intrinsic_rewards)))), av))
                    else:
                        av = sum(self.intrinsic_rewards[low:high]) / (high - low)
                        f.write("[{}, {}) {} {} \n".format(low, high, " " * (25 - 2 - len(str(low)) - len(str(high))), av))

            # print table of x distance walked over session
            f.write("\n\nAverage x distance travelled over past {} episodes: \n".format(len(self.distance_array)))
            f.write("EPISODE RANGE                AV. DISTANCE \n")
            section_size = math.floor(len(self.distance_array) / sections)

            for i in range(sections):
                low = i * section_size
                high = (i + 1) * section_size

                if i == sections - 1:
                    av = sum(self.distance_array[low:]) / (len(self.distance_array) - low)
                    f.write("[{}, {}) {} {} \n".format(low, len(self.distance_array), " " * (25 - 2 - len(str(low)) - len(str(len(self.distance_array)))), av))
                else:
                    av = sum(self.distance_array[low:high]) / (high - low)
                    f.write("[{}, {}) {} {} \n".format(low, high, " " * (25 - 2 - len(str(low)) - len(str(high))), av))

            # print table of training times for each timestep over session
            f.write("\n\nAverage training time per time-step over past {} time-steps: {}".format(len(self.training_times), sum(self.training_times) / len(self.distance_array)))
