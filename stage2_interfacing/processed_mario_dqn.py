import torch
import numpy as np
import pickle  # util for converting between python object structures and byte streams

import torch.nn as nn
import random


class Network(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Network, self).__init__()
        # convolutional layers with increasing width and decreasing kernel size & stride
        self.conv = nn.Sequential(
            # set up input layer size from input state shape, here [4, 84, 84] i.e. 4 frames of 84x84 pixels
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            # use ReLU as activation function
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # retrieve output size of convolutional set by measuring the size generated by a zeroed input
        pss = self.conv(torch.zeros(1, *input_shape))
        conv_out_size = int(np.prod(pss.size()))

        # linear layers preparing output to desired size n_actions (one output neuron per element of action vector)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    # forward pass combining conv set and lin set
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class Agent:
    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr,
                 exploration_max, exploration_min, exploration_decay, pretrained):

        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.local_net = Network(state_space, action_space).to(self.device)
        self.target_net = Network(state_space, action_space).to(self.device)

        if self.pretrained:
            self.local_net.load_state_dict(torch.load("dq1.pt", map_location=torch.device(self.device)))
            self.target_net.load_state_dict(torch.load("dq2.pt", map_location=torch.device(self.device)))

        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
        self.copy = 5000  # Copy the local model weights into the target network every 5000 steps
        self.step = 0

        # Create memory
        self.max_memory_size = max_memory_size
        if self.pretrained:
            self.STATE_MEM = torch.load("STATE_MEM.pt")
            self.ACTION_MEM = torch.load("ACTION_MEM.pt")
            self.REWARD_MEM = torch.load("REWARD_MEM.pt")
            self.STATE2_MEM = torch.load("STATE2_MEM.pt")
            self.DONE_MEM = torch.load("DONE_MEM.pt")
            with open("ending_position.pkl", 'rb') as f:
                self.ending_position = pickle.load(f)
            with open("num_in_queue.pkl", 'rb') as f:
                self.num_in_queue = pickle.load(f)
        else:
            self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0

        self.memory_sample_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device)  # Also known as Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    # pushing an experience tuple (transition) to replay memory (fixed length deque)
    def remember(self, state, action, reward, state2, done):
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def recall(self):
        # Randomly sample 'batch size' experiences
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)

        STATE = self.STATE_MEM[idx].to(self.device)
        ACTION = self.ACTION_MEM[idx].to(self.device)
        REWARD = self.REWARD_MEM[idx].to(self.device)
        STATE2 = self.STATE2_MEM[idx].to(self.device)
        DONE = self.DONE_MEM[idx].to(self.device)

        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state):
        # Epsilon-greedy action

        self.step += 1

        if random.random() < self.exploration_rate:
            print("Random action taken")
            return torch.tensor([[random.randrange(self.action_space)]])

        # Local net is used for the policy

        nn_out = self.local_net(state.to(self.device))
        nn_out = torch.argmax(nn_out).unsqueeze(0).unsqueeze(0).cpu()

        print("Neural net output = ", nn_out)

        return nn_out

    def copy_model(self):
        # Copy local net weights into target net

        self.target_net.load_state_dict(self.local_net.state_dict())

    # conducting experience replay training
    def experience_replay(self):

        # updating target network at regular intervals
        if self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return

        # sample transition
        STATE, ACTION, REWARD, STATE2, DONE = self.recall()

        # set gradients of tensors to zero
        self.optimizer.zero_grad()

        # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
        target = REWARD + torch.mul((self.gamma *
                                     self.target_net(STATE2).max(1).values.unsqueeze(1)),
                                    1 - DONE)

        # Local net approximation of Q-value
        current = self.local_net(STATE).gather(1, ACTION.long())

        # compute gradients to minimise loss between policy approximation and target, then conduct backpropagation step
        loss = self.l1(current, target)
        loss.backward()
        self.optimizer.step()

        # decay epsilon value
        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)