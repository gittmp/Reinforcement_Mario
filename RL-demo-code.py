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


class Buffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=max_buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, size):
        batch = random.sample(self.buffer, size)
        Ss, As, Rs, Succs, Terms = [], [], [], [], []

        for transition in batch:
            s, a, r, succ, term = transition
            Ss.append(s)
            As.append([a])
            Rs.append([r])
            Succs.append(succ)
            Terms.append([term])

        S_tensor = torch.tensor(Ss, dtype=torch.float)
        A_tensor = torch.tensor(As)
        R_tensor = torch.tensor(Rs)
        Succ_tensor = torch.tensor(Succs, dtype=torch.float)
        Term_tensor = torch.tensor(Terms)

        return S_tensor, A_tensor, R_tensor, Succ_tensor, Term_tensor

    def size(self):
        return len(self.buffer)


class Network(nn.Module):
    def __init__(self, input_size, output_size):
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

    def step(self, inputs, epsilon):
        outputs = self.forward(inputs)

        exploration_probability = random.random()
        if exploration_probability < epsilon:
            return random.randint(0, 1)
        else:
            return outputs.argmax().item()


def train_step(optimiser_network, target_network, memory, optimser):
    s, a, r, succ, terminal = memory.sample(batch_size)

    qvalues = optimiser_network(s)
    q_sa = qvalues.gather(1, a)

    succ_qvalues = target_network(succ)
    max_q_succ = succ_qvalues.max(1)[0].unsqueeze(1)
    q_target = r + gamma * max_q_succ * terminal

    loss = funt.smooth_l1_loss(q_sa, q_target)
    optimser.zero_grad()
    loss.backward()
    optimser.step()


env = gym.make('CartPole-v0')
env.reset()

# hyperparameters

alpha = 0.0005
gamma = 0.98
max_buffer_size = 50000
batch_size = 32

# seed behaviour of spaces such that they are reproducible
seed = 742
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
env.seed(seed)
env.action_space.seed(seed)

state_space_size = np.array(env.observation_space.shape).prod()
action_space_size = env.action_space.n
optimising_network = Network(state_space_size, action_space_size)
target_network = Network(state_space_size, action_space_size)
target_network.load_state_dict(optimising_network.state_dict())
replay_buffer = Buffer()

no_eps = 3000
min_epsilon = 0.01
max_epsilon = 0.08
epsilon_anneal_rate = 0.01/200
training_init_size = 2000
ep_rewards = []
marking = []
means = []
printing_rate = 50
marking_rate = 50
optimiser = optim.Adam(optimising_network.parameters(), lr=alpha)

for ep in range(no_eps):
    epsilon = max(min_epsilon, max_epsilon - epsilon_anneal_rate * ep)
    state = env.reset()
    total_reward = 0.0
    time_step = 0

    while True:
        time_step += 1
        action = optimising_network.step(torch.from_numpy(state).float().unsqueeze(0), epsilon)
        successor, reward, terminal, info = env.step(action)

        terminal_mask = 0.0 if terminal else 1.0
        transition = (state, action, reward/100.0, successor, terminal_mask)
        replay_buffer.add(transition)
        state = successor
        total_reward += reward

        if terminal:
            break

    if replay_buffer.size() > training_init_size:
        train_step(optimising_network, target_network, replay_buffer, optimiser)

    ep_rewards.append(total_reward)
    marking.append(total_reward)
    if ep % marking_rate == 0:
        mean = np.array(marking).mean()
        means.append(mean)
        print("marking, episode: {}, total reward: {:.1f}, mean score: {:.2f}, std score: {:.2f}".format(ep, total_reward, mean, np.array(marking).std()))
        marking = []

    if ep % printing_rate == 0 and ep != 0:
        target_network.load_state_dict(optimising_network.state_dict())
        print("episode: {}, total reward: {:.1f}, epsilon: {:.2f}".format(ep, total_reward, epsilon))

# no_sections = 100
# section_size = int(no_eps/no_sections)
# averages = []
# for i in range(no_sections):
#     low = i*section_size
#     high = (i+1)*section_size
#     avg = sum(ep_rewards[low:high]) / section_size
#     averages.append(avg)

# Plot the reward array showing the progression of the agents success
# reward_samples = [ep_rewards[i] for i in range(len(ep_rewards)) if i % 50 == 0]
plt.plot(means, color='red')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training progression')
plt.show()
