# SOURCE: https://colab.research.google.com/gist/qazwsxal/6cc1c5cf16a23ae6ea8d5c369828fa80/gym-demo.ipynb
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
    def __init__(self):
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
        if expl_probability < eps:
            return random.randint(0, 1)
        else:
            return outputs.argmax().item()


def train_step(optimising_net, target_net, memory, optimser, ep_no):
    s, a, r, succ, term = memory.sample(batch_size)

    qvalues = optimising_net(s)
    q_sa = qvalues.gather(1, a)

    # generate q values for all actions from the successor state
    succ_qvalues = target_net(succ)
    # generate tensor of the maximal action for the successor state for each element in the batch
    max_q_succ = succ_qvalues.max(1)[0].unsqueeze(1)
    # Q-target = reward + discounted maximal value of successor state (over all actions)
    # if terminal, just reward (mask out rest)
    q_target = r + gamma * max_q_succ * term

    loss = funt.smooth_l1_loss(q_sa, q_target)
    optimser.zero_grad()
    loss.backward()
    optimser.step()

    # at regular intervals (every 50 episodes) bring target network up to date with optimising network
    if ep_no % 10 == 0 and ep != 0:
        target_net.load_state_dict(optimising_net.state_dict())


# initialise environment
env = gym.make('CartPole-v0')
start = time.time()

# network hyperparameters
alpha = 0.005
gamma = 0.98
max_buffer_size = 50000
batch_size = 32
min_epsilon = 0.01
max_epsilon = 0.08
epsilon_anneal_rate = 0.01/200
training_init_size = 1500

# seed behaviour of spaces such that they are reproducible
seed = 742
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
env.seed(seed)
env.action_space.seed(seed)

# generate replay buffer and agent's two networks: optimising and target (init target params equal to optimising ones)
state_space_size = np.array(env.observation_space.shape).prod()
action_space_size = env.action_space.n
optimising_network = Network(state_space_size, action_space_size)
target_network = Network(state_space_size, action_space_size)
target_network.load_state_dict(optimising_network.state_dict())
replay_buffer = Buffer()

no_eps = 2000
marking = []
means = []
printing_rate = 50
marking_rate = 25
optimiser = optim.Adam(optimising_network.parameters(), lr=alpha)

for ep in range(no_eps):
    epsilon = max(min_epsilon, max_epsilon - epsilon_anneal_rate * ep)
    state = env.reset()
    total_reward = 0.0
    time_step = 0

    while True:
        time_step += 1
        action = optimising_network.step(torch.from_numpy(state).float().unsqueeze(0), epsilon)
        successor, reward, terminal, _ = env.step(action)

        terminal_mask = 0.0 if terminal else 1.0
        transition = (state, action, reward/100.0, successor, terminal_mask)
        replay_buffer.add(transition)
        state = successor
        total_reward += reward

        if terminal:
            break

    if replay_buffer.size() > training_init_size:
        train_step(optimising_network, target_network, replay_buffer, optimiser, ep)

    marking.append(total_reward)
    if ep % marking_rate == 0:
        std = np.array(marking).std()
        mean = np.array(marking).mean()
        means.append(mean)
        print("marking, episode: {}, total reward: {:.1f}, mean score: {:.2f}, std score: {:.2f}"
              .format(ep, total_reward, mean, std))
        marking = []

    # if ep % printing_rate == 0 and ep != 0:
        # print("episode: {}, total reward: {:.1f}, epsilon: {:.2f}".format(ep, total_reward, epsilon))

enlapsed = time.time() - start
print("Time elapsed: ", enlapsed)

# Plot the mean reward array showing the progression of the agents success
plt.plot(means, color='red')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training progression [time = {:.2f}]'.format(enlapsed))
plt.savefig('training_progression_{}.png'.format(datetime.datetime.now()))
plt.show()
