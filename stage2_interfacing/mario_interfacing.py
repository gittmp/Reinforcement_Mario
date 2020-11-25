# Need OpenGL for this: pip3 install PyOpenGL PyOpenGL_accelerate
# takes input of section of screen frames to produce q-values
# state represented as difference between current frame and last one
# this allows agent to deduce the velocity

# imports
import gym
import random
import numpy as np
import torch
import torch.nn as nn  # nn package
import torch.nn.functional as funct
import torch.optim as optim  # nn optimisation
import matplotlib.pyplot as plt

import math
import matplotlib
from collections import namedtuple
from itertools import count
from PIL import Image
import torchvision.transforms as trans  # utility for computer vision

import retro


# make environment
# env = gym.make('CartPole-v0').unwrapped
env = retro.make('SuperMarioBros-Nes').unwrapped
plt.ion()

print("State space: ", env.observation_space)
# print("State space high: ", env.observation_space.high)
# print("State space low: ", env.observation_space.low)
print("Action space: ", env.action_space)

# use GPU if possible, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# build replay memory to sample transitions from randomly - stabilises and improves training
# transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# cyclic buffer of bounded size storing transitions
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# convolutional neural network model
# input = difference between current and last frame subsections
# output = array of 2 q-values, one for each action (left and right)
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        # 1st conv layer of 3 input channels and 16 output channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        # batch normalisation on outputs of previous convolutional layer
        self.bn1 = nn.BatchNorm2d(16)
        # 2 more convolutional layers and their associated normalisations
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # size of linear input layer (no.connections) depends on the output size of the conv layers
        def conv2d_output_size(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_w = conv2d_output_size(conv2d_output_size(conv2d_output_size(w)))
        conv_h = conv2d_output_size(conv2d_output_size(conv2d_output_size(h)))

        # input layer must be of size: width = conv_w, height = conv_h, depth = batch_size = 32
        linear_input_size = conv_w * conv_h * 32

        # construct the input layer (head)
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        # forward pass through network to produce q-values for a state / batch of states
        # inputs are first passed into conv layer, then normalised, then have the activation function ReLU applied
        x = funct.relu(self.bn1(self.conv1(x)))
        x = funct.relu(self.bn2(self.conv2(x)))
        x = funct.relu(self.bn3(self.conv3(x)))
        # return output as tensor([[q_0,left, q_0,right] ... ]) for batch
        out = self.head(x.view(x.size(0), -1))
        return out


# input extraction
# extraction/processing of images rendered from the environment
# displays extracted subsection when cell is run
# start with resizing function that takes a frame and generates a PIL image, resized to 40^2, represented as a tensor
resize = trans.Compose([
    trans.ToPILImage(),
    trans.Resize(40, interpolation=Image.CUBIC),
    trans.ToTensor()
])


# find location of agent within the frame
def get_cart_location(screen_width):
    # width of whole environment is length of +ve x-axis doubled (to incl. -ve)
    world_width = env.x_threshold * 2  # ERROR this line wont work as no x_threshold attribute
    # scaling factor is ratio between screen width and world width
    scale = screen_width / world_width
    # find the point of the middle of the cart
    cart_loc = int(env.state[0] * scale + screen_width / 2.0)
    return cart_loc


def get_screen():
    # transform screen returned from gym into torch order (CHW)
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    """
    # strip redundant vertical top 20% and bottom 40% of image (as cart not there)
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    # set desired width of viewing areas to be 60% of the screen
    view_width = int(screen_width * 0.6)

    # cut away redundant horizontal areas of screen without cart in
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        # if cart is in first 1/3 of screen, keep slice 0 -> |view_width|
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        # if cart in last 1/3 of screen, keep slice -|view_width| -> end (wraps around to keep last |view_width| values)
        slice_range = slice(-view_width, None)
    else:
        # otherwise (in middle 1/3) keep the |view_width| pixels around the cart (|view_width|/2 left and same right)
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    # strip screen in horizontal plane s.t. only the slice_range is left
    screen = screen[:, :, slice_range]
    """
    # convert screen array to float and rescale as a continuous array.
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    # convert this into a torch tensor
    screen = torch.from_numpy(screen)
    # resize and add dimension to hold others in the batch (BCHW)
    return resize(screen).unsqueeze(0).to(device)


env.reset()
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
# plt.title('Example extracted screen')
# plt.show()

batch_size = 128
gamma = 0.999
epsilon_ceil = 0.9
epsilon_floor = 0.05
epsilon_decay = 200
update_target = 20

# get screen size/shape to initialised input layer size of NN
init_screen = get_screen()
_, _, height, width = init_screen.shape

# get action space size
n_actions = env.action_space.n

# initialise policy network and target network
policy_network = DQN(height, width, n_actions).to(device)
target_network = DQN(height, width, n_actions).to(device)
target_network.load_state_dict(policy_network.state_dict())
target_network.eval()

# initialise optimiser and replay memory
optimiser = optim.RMSprop(policy_network.parameters())
memory = ReplayMemory(10000)

time_step = 0


# step function picking action from current state
def select_action(state):
    global time_step
    epsilon = epsilon_floor + (epsilon_ceil - epsilon_floor) * math.exp(-1.0 * time_step / epsilon_decay)
    time_step += 1
    if random.random() > epsilon:
        with torch.no_grad():
            # t.max(1) = return max column value of each row
            # [1] = return index of that max
            # i.e. return action with the largest value
            return policy_network(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


# plot function which plots durations the figure during training.
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training')
    # plt.xlabel('Episode')
    plt.xlabel('Timestep')
    # plt.ylabel('Duration')
    plt.ylabel('Reward')
    plt.plot(durations_t.numpy())

    # plot 100 means of episodes
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # update plots
    plt.pause(0.001)


# training model over one step using a batch
def optimise_model():
    if len(memory) < batch_size:
        return

    # sample transitions and convert batch-array of Transitions to a Transition of batch-arrays
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # mask states s.t. terminal states don't have a successor to eval
    non_term_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)

    # concatenate batch elements into tensors of successors, states, actions and rewards
    non_term_successors = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # compute Q(s,a) via policy (optimising) network
    # evaluate all successor states for use in TD-target (via target net)
    action_vals = policy_network(state_batch).gather(1, action_batch)
    successor_vals = torch.zeros(batch_size, device=device)
    successor_vals[non_term_mask] = target_network(non_term_successors).max(1)[0].detach()
    td_targets = (successor_vals * gamma) + reward_batch

    # compute huber loss between values and targets
    loss = funct.smooth_l1_loss(action_vals, td_targets.unsqueeze(1))

    # conduct backpropagation
    optimiser.zero_grad()
    loss.backward()
    for param in policy_network.parameters():
        param.grad.data.clamp_(-1, 1)
    optimiser.step()


# main loop of agents interactions and experience replay training
num_eps = 300
for ep in range(num_eps):
    # represent state as the difference between the last two frames of gameplay
    print("Episode: ", ep)
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen

    # tick through time-steps
    for t in count():
        print("Time-step =", t)
        action = select_action(state)
        # _, reward, terminal, _ = env.step(action.item())
        _, reward, terminal, _ = env.step(env.action_space.sample())
        reward = torch.tensor([reward], device=device)

        # print("Actual state: ", state)
        # print("Random state: ", env.observation_space.sample())

        # print("NN action: ", action)
        # print("Random action: ", env.action_space.sample())

        last_screen = current_screen
        current_screen = get_screen()

        if not terminal:
            successor = current_screen - last_screen
        else:
            successor = None

        # push to replay buffer
        memory.push(state, action, successor, reward)
        state = successor

        # episode_durations.append(reward)
        # plot_durations()
        env.render()

        # optimise policy network at every step
        optimise_model()
        if terminal:
            # episode_durations.append(t + 1)
            # plot_durations()
            break

    if ep % update_target == 0:
        target_network.load_state_dict(policy_network.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
