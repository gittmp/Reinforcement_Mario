import torch
import random
import numpy as np
from collections import deque


# experience replay memory
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


# sum tree utilised for priorities experience replay
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
            return None, None, None

        return idx, float(self.tree[idx]), exp

    def total(self):
        return float(self.tree[0])

    def full(self):
        return self.length >= self.maxlen

    # a new experience with its associated priority
    def push(self, pri, exp):
        ind = self.pointer + self.maxlen - 1

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


# prioritised experience replay
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

            if index is None:
                return None, None, None

            priorities[i] = priority
            indices[i] = index

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