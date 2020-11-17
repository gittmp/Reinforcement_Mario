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


# make environment
env = gym.make('CartPole-v0').unwrapped
# plt.ion()

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
        out = self.head(x.view(x.size(), -1))
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
    world_width = env.x_threshold * 2
    # scaling factor is ratio between screen width and world width
    scale = screen_width / world_width
    # find the point of the middle of the cart
    cart_loc = int(env.state[0] * scale + screen_width / 2.0)
    return cart_loc


def get_screen():
    # transform screen returned from gym into torch order (CHW)
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
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
    # convert screen array to float and rescale as a continuous array.
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    # convert this into a torch tensor
    screen = torch.from_numpy(screen)
    # resize and add dimension to hold others in the batch (BCHW)
    return resize(screen).unsqueeze(0).to(device)



env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
plt.title('Example extracted screen')
plt.show()
