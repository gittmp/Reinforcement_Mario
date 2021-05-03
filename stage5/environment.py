import gym_super_mario_bros as SMBGym
import numpy as np
from nes_py.wrappers import JoypadSpace  # converting actions to correct JoyPad representation
from collections import deque  # for deque: insert(e_new) -> [e_new, e_0, ..., e_l-2] -> exit(e_l-1)
import gym
import cv2  # util for image manipulation

from utilities import *


# CLASSES / FUNCTIONS FOR WRAPPING ENVIRONMENT TO IMPROVE PERFORMANCE
# frame-skipping, and intrinsic reward system
class SkipAndReward(gym.Wrapper):
    # initialise no. frames to skip and frame buffer as deque length 2
    def __init__(self, env=None, path=None, version=2, skip=4):
        super(SkipAndReward, self).__init__(env)

        self.env = env
        self.k = skip
        self.frame_buffer = deque(maxlen=2)
        self.x_buffer = deque(maxlen=8)
        self.prev_score = 0
        self.prev_grad = 1
        self.zero_grad_counter = 0
        self.path = path
        self.version = version

    # calculate the gradient of change of position over the position buffer
    def x_gradient(self):
        entries = len(self.x_buffer)
        gradient = 0

        # average change over all adjacent states in buffer
        for i in range(1, entries):
            x2 = self.x_buffer[i - 1]
            x1 = self.x_buffer[i]
            gradient += x1 - x2

        gradient = round(gradient / entries)

        # give corresponding reward (decay below zero if continued non-positive gradient)
        if gradient <= 0 and self.prev_grad <= 0:
            gradient = self.zero_grad_counter * -0.1 + gradient
            self.zero_grad_counter += 1
        else:
            self.prev_grad = gradient
            self.zero_grad_counter = 0

        return gradient

    def modify_reward(self, rew, data):
        # position reward
        self.x_buffer.append(data['x_pos'])
        x_reward = self.x_gradient()

        # extrinsic reward
        score_reward = data['score'] - self.prev_score
        self.prev_score = data['score']

        # status reward
        if data['status'] == 'fireball':
            status_reward = 10
        elif data['status'] == 'tall':
            status_reward = 10
        else:
            status_reward = 0

        reward = round(x_reward + rew + score_reward + status_reward)

        # normalise to range [-15, 15]
        if reward > 15 or data['flag_get']:
            reward = 15
        if reward < -15:
            reward = -15

        return reward

    # override step method to conduct frame-skipping according to parameter k
    def step(self, action):
        if self.version == 0:
            frame, total_reward, terminal, info = self.env.step(action)
        else:  # self.version in [1, 2]
            total_reward = 0.0
            terminal = None
            info = -1

            # skip next k frames
            for _ in range(self.k):
                state, reward, terminal, info = self.env.step(action)

                if self.version == 2:
                    reward = self.modify_reward(reward, info)

                self.frame_buffer.append(state)
                total_reward += reward

                # if reach the end of an episode, escape and leave obs buffer containing only the last 2 frames seen
                if terminal:
                    break

            # stack elements of obs buffer = create new array containing the two states in the obs buffer
            # then take the max of this, to return the maximal state
            frame = np.max(np.stack(self.frame_buffer), axis=0)

        # return this maximum frame, the total reward generated over the `skip` frames, and whether the game has ended
        return frame, total_reward, terminal, info

    # override reset method so that all internal vars are also reset
    def reset(self):
        self.frame_buffer.clear()
        self.x_buffer.clear()
        self.prev_score = 0
        state = self.env.reset()
        self.frame_buffer.append(state)

        return state


# Wrapper (for observation space) to down-sample frame to numpy array of 104x140 greyscale pixels
class ProcessFrame(gym.ObservationWrapper):
    # represent the observation space as a box of 104x140 pixel values with only 1 colour channel (greyscale)
    def __init__(self, env=None, version=2, new_shape=(104, 140, 1), old_shape=(240, 256, 3)):
        super(ProcessFrame, self).__init__(env)
        self.env = env
        self.version = version
        self.old_shape = [old_shape[0], old_shape[1], old_shape[2]]
        self.new_shape = new_shape

        if version == 0:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=old_shape, dtype=np.uint8)
        else:  # self.version in [1, 2]
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    # override observation method s.t. it returns the downsampled version
    def observation(self, obs):
        if self.version == 0:
            return np.reshape(obs, self.old_shape).astype(np.uint8)
        else:  # self.version in [1, 2]
            # generate new pixel frame
            return ProcessFrame.process(self, obs)

    # downsampling process
    def process(self, frame):
        # convert to numpy of originally shaped 240x256 pixel values of 3 colour channels (RGB)
        img = np.reshape(frame, self.old_shape).astype(np.float32)

        # convert to 1 channel greyscale image by taking 29.9% of R channel + 58.7% G + 11.4% B
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

        # downscale frame using INTER AREA
        # INTER AREA = resamples input image using pixel area relation
        resized_screen = cv2.resize(img, (self.new_shape[1], self.new_shape[1]), interpolation=cv2.INTER_AREA)

        # crop to new shape by cutting off redundant height
        crop_factor = int((self.new_shape[1] - self.new_shape[0]) / 2)
        output = resized_screen[crop_factor:self.new_shape[1] - crop_factor, :]
        output = np.reshape(output, [self.new_shape[0], self.new_shape[1], self.new_shape[2]]).astype(np.uint8)

        return output


# adapt state space to reflect changes made during downscaling
class CorrectStateSpace(gym.ObservationWrapper):
    def __init__(self, env, version=2):
        super(CorrectStateSpace, self).__init__(env)
        self.env = env
        old_shape = self.observation_space.shape

        if version == 0:
            self.observation_space = gym.spaces.Box(low=0.0,
                                                    high=255.0,
                                                    shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                    dtype=np.float32)
        else:  # self.version in [1, 2]
            # reshape frame such that it is in the correct format for the state space
            # normalise to range [0, 1] for neural network
            self.observation_space = gym.spaces.Box(low=0.0,
                                                    high=1.0,
                                                    shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                    dtype=np.float32)

    def observation(self, observation):
        out = np.moveaxis(observation, 2, 0)
        return out


# adapt state space to reflect changes made during frame-skipping
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, version=2):
        super(BufferWrapper, self).__init__(env)
        self.env = env
        self.version = version
        old_space = env.observation_space
        self.dtype = np.float32
        self.observation_space = gym.spaces.Box(old_space.low,
                                                old_space.high,
                                                dtype=self.dtype)

    # reset state buffer
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    # append new state to the state buffer
    def observation(self, observation):
        if self.version == 0:
            return np.array(observation).astype(np.float32)
        else:  # self.version in [1, 2]
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = observation

            # normalise states to range [0, 1]
            return np.array(observation).astype(np.float32) / 255.0


# apply pipeline of state/action space manipulations
def make_env(game, path, version):
    # make env
    env = SMBGym.make(game)

    # apply pipeline
    env = SkipAndReward(env, path, version)
    env = ProcessFrame(env, version)
    env = CorrectStateSpace(env, version)
    env = BufferWrapper(env, version)

    # if not version 0, transform action space
    if version in [1, 2]:
        env = JoypadSpace(env, ACTION_SET)

    return env

