# adaptations to the gym_super_mario_bros environment found at https://github.com/Kautenja/gym-super-mario-bros
import gym_super_mario_bros as SMBGym
import numpy as np
from nes_py.wrappers import JoypadSpace  # converting actions to correct JoyPad representation
from collections import deque  # for deque: insert(e_new) -> [e_new, e_0, ..., e_l-2] -> exit(e_l-1)
import gym
import cv2  # util for image manipulation


# Limiting action set combinations:
# actions for the simple run right environment
# right + B = run; hold A = jump higher; right + A + B = running jump;
# B = throw fire; water A = bob; down = crouch; start = play/pause;
RIGHT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
]

# actions for very simple movement
SIMPLE = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left']
]

# actions for more complex movement
COMPLEX = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]

# Custom action combination
ACTION_SET = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A']
]


# CLASSES / FUNCTIONS FOR WRAPPING ENVIRONMENT TO IMPROVE PERFORMANCE
# Wrapper (for whole environment) which return only every `skip`-th frame, and adapts reward function
class SkipAndReward(gym.Wrapper):
    # initialise no. frames to skip and frame buffer as deque length 2
    def __init__(self, env=None, skip=4):
        super(SkipAndReward, self).__init__(env)
        self.env = env
        self._skip = skip
        self.frame_buffer = deque(maxlen=2)
        self.x_buffer = deque(maxlen=7)
        self.reward_buffer = []
        self.prev_score = 0
        self.prev_status = 'small'
        self.prev_grad = 1
        self.zero_grad_counter = 0

    def x_gradient(self):
        entries = len(self.x_buffer)
        gradient = 0

        for i in range(1, entries):
            x2 = self.x_buffer[i - 1]
            x1 = self.x_buffer[i]
            gradient += x1 - x2

        gradient = round(gradient / entries)

        if gradient == 0 and self.prev_grad == 0:
            gradient = self.zero_grad_counter * -0.1
            self.zero_grad_counter += 1
        else:
            self.prev_grad = gradient
            self.zero_grad_counter = 0

        return gradient

    def modify_reward(self, rew, data):
        # rew = v + c + d
        # v = difference in x positions [-15, 15]; c = difference in the game clock [-15, 0]; d = death penalty {-15, 0}

        # get average gradient of x change in last 4 moves
        self.x_buffer.append(data['x_pos'])
        x_reward = self.x_gradient()

        if data['flag_get']:
            flag_reward = 15
        else:
            flag_reward = 0

        score_reward = data['score'] - self.prev_score
        self.prev_score = data['score']

        if data['status'] == 'fireball' and self.prev_status != 'fireball':
            status_reward = 6
            self.prev_status = 'fireball'
        elif data['status'] == 'tall' and self.prev_status != 'tall':
            status_reward = 6
            self.prev_status = 'tall'
        else:
            status_reward = 0
            self.prev_status = 'small'

        reward = round(x_reward + abs(rew) + flag_reward + score_reward + status_reward)

        if reward > 15:
            reward = 15
        if reward < -15:
            reward = -15

        self.reward_buffer.append(reward)

        return reward

    # override step method to go forward `skip` frames after picking `action` in current frame
    def step(self, action):
        total_reward = 0.0
        terminal = None
        info = -1

        # step through next `skip` frames of gameplay
        for _ in range(self._skip):
            state, reward, terminal, info = self.env.step(action)
            # reward = self.modify_reward(reward, info)
            self.frame_buffer.append(state)
            total_reward += reward

            # if reach the end of an episode, escape and leave obs buffer containing only the last 2 frames seen
            if terminal:
                break

        # stack elements of obs buffer = create new array containing the two states in the obs buffer
        # then take the max of this, to return the maximal state (of the last two remaining in the buffer)
        max_frame = np.max(np.stack(self.frame_buffer), axis=0)

        # return this maximum frame, the total reward generated over the `skip` frames, and whether the game has ended
        return max_frame, total_reward, terminal, info

    # override reset method so that it also resets the internal obs buffer to only hold the initial state
    def reset(self):
        length = len(self.reward_buffer)
        if length > 0:
            print("Mean reward over last {} time-steps = {:.1f}\n".format(length, sum(self.reward_buffer) / length))

        self.reward_buffer = []
        self.frame_buffer.clear()
        self.x_buffer.clear()
        self.prev_score = 0
        self.prev_status = 'small'

        state = self.env.reset()
        self.frame_buffer.append(state)

        return state


# Wrapper (for observation space) to down-sample frame to numpy array of 84x84 greyscale pixels
class ProcessFrame(gym.ObservationWrapper):
    # represent the observation space as a box of 84x84 pixel values with only 1 colour channel (greyscale)
    def __init__(self, env=None, new_shape=(104, 140, 1), old_shape=(240, 256, 3)):
        super(ProcessFrame, self).__init__(env)
        self.env = env
        self.old_shape = [old_shape[0], old_shape[1], old_shape[2]]
        self.new_shape = new_shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    # override observation method s.t. it returns the down-sampled version
    def observation(self, obs):
        # generate 84x84 pixel frame
        return ProcessFrame.process(self, obs)

    # form static method bound to the class itself to handle the process of generating the down-sampled frames
    def process(self, frame):
        # form numpy array of gameplay frame in original shape of 240x256 pixel values of 3 colour channels (RGB)
        img = np.reshape(frame, self.old_shape).astype(np.float32)

        # convert to 1 channel greyscale image by taking 29.9% of R channel + 58.7% G + 11.4% B
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

        # down-scale frame to 1 channel 84x110 pixel values using INTER AREA
        # INTER AREA = resamples input image using pixel area relation
        resized_screen = cv2.resize(img, (self.new_shape[1], self.new_shape[1]), interpolation=cv2.INTER_AREA)

        # reshape to square 84x84 numpy array (of 8-bit unsigned integers) by cutting off redundant height
        crop_factor = int((self.new_shape[1] - self.new_shape[0]) / 2)
        output = resized_screen[crop_factor:self.new_shape[1] - crop_factor, :]
        output = np.reshape(output, [self.new_shape[0], self.new_shape[1], self.new_shape[2]]).astype(np.uint8)

        return output


# Wrapper (for observation space) converting the greyscale 84x84 image to a pytorch tensor with scaled values 0-1
class ImageToPyTorch(gym.ObservationWrapper):
    # change obs space representation to reflect values in range 0-1 (shape also changed s.t. 3rd dimension is now 1st)
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        self.env = env
        old_shape = self.observation_space.shape

        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    # override observation method s.t. it returns in new shifted configuration
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


# observation buffer wrapper
class BufferWrapper(gym.ObservationWrapper):
    # redefine obs space box to represent the fact its holding 4 frames (4 identical low vals, 4 identical high vals)
    def __init__(self, env, n_steps=4, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.env = env
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0),
                                                dtype=dtype)

    # override reset method to also reset memory buffer to the same shape as obs low vals but all elements are 0
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    # move the elements of memory buffer 1 index forward (removing 1st), and set the last item as the new observation
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation

        # normalise elements to an np array of floats in the range 0-1 (by dividing through by max 8-bit pixel value)
        output = np.array(self.buffer).astype(np.float32) / 255.0
        return output


# Function combining into pipeline of wrapper transforms
def make_env(game):
    # make env
    env = SMBGym.make(game)

    # apply pipeline
    env = SkipAndReward(env)
    env = ProcessFrame(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env)
    env = JoypadSpace(env, ACTION_SET)

    return env

