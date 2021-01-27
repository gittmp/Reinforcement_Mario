# MARIO GYM ENV: https://github.com/Kautenja/gym-super-mario-bros

import os
import torch
import torchvision
import gym_super_mario_bros as gym_smb
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt

import mario_wrapper as wrapper
import mario_network as network

# import retro
import time


# plot function which plots durations the figure during training.
def plot_durations(ep_rewards):
    plt.figure(2)
    plt.clf()
    rewards = torch.tensor(ep_rewards, dtype=torch.float)
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards.numpy())

    # plot 100 means of episodes
    if len(rewards) >= 100:
        means = rewards.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # update plots
    plt.pause(0.001)


# plot function to visualise downsampled slice of screen
def render_state(four_frames):
    single_image = four_frames.squeeze(0)[-1]
    fig = plt.figure("Frame")
    plt.imshow(single_image)
    plt.title("Down-sampled 84x84 grayscale image")
    plt.draw()
    plt.pause(0.001)


game = 'SuperMarioBros-1-1-v0'
env = gym_smb.make(game)

# game = "SuperMarioBros-Nes"
# env = retro.make(game).unwrapped

print("PRE-WRAPPING")
print("obs space: ", env.observation_space)
print("action space:", env.action_space)
print("sample action: ", env.action_space.sample())

env = wrapper.make_env(env)

print("\nPOST-WRAPPING")
print("obs space: ", env.observation_space)
print("action space:", env.action_space)
print("sample action: ", env.action_space.sample())

pretrained = os.path.isfile("params/episode_rewards.pkl")

agent = network.Agent(
    state_shape=env.observation_space.shape,
    action_n=env.action_space.n,
    alpha=0.00025,
    gamma=0.9,
    epsilon_ceil=1.0,
    epsilon_floor=0.02,
    epsilon_decay=0.99,
    buffer_capacity=30000,
    batch_size=32,
    update_target=5000,
    pretrained=pretrained
)

if pretrained:
    with open("params/episode_rewards.pkl", "rb") as f:
        episode_rewards = pickle.load(f)
else:
    episode_rewards = []

training = True
no_eps = 25
env.reset()

# render_times = []
action_times = []
environment_times = []
train_times = []

for ep in tqdm(range(no_eps)):
    state = env.reset()
    state = torch.Tensor([state])
    total_reward = 0
    timestep = 0

    # render_times_av = []
    action_times_av = []
    environment_times_av = []
    train_times_av = []
    plot_times_av = []

    while True:
        timestep += 1

        # start = time.time()
        #
        env.render()

        if timestep % 10 == 0:
            render_state(state)
        #
        # enlapsed = time.time() - start
        # render_times_av.append(enlapsed)
        start = time.time()

        action = agent.step(state)

        enlapsed = time.time() - start
        action_times_av.append(enlapsed)
        start = time.time()

        # sample_action = torch.Tensor(env.action_space.sample())
        # print("Chosen action = ", action)
        # print("Sampled action = ", sample_action)

        successor, reward, terminal, info = env.step(int(action[0]))
        total_reward += reward

        enlapsed = time.time() - start
        environment_times_av.append(enlapsed)

        successor = torch.Tensor([successor])
        reward = torch.Tensor([reward]).unsqueeze(0)
        terminal = torch.Tensor([int(terminal)]).unsqueeze(0)

        if training:
            start = time.time()

            experience = (state.float(), action.float(), reward.float(), successor.float(), terminal.float())
            agent.memory.push(experience)
            agent.train()

            enlapsed = time.time() - start
            train_times_av.append(enlapsed)

        state = successor

        if terminal:
            print("\nInfo:\nfinal game score = {}, time elapsed = {}, Mario's location = ({}, {})"
                  .format(info['score'], 400 - info['time'], info['x_pos'], info['y_pos']))
            break

    episode_rewards.append(total_reward)
    print("\nTotal reward after episode {} is {}".format(ep + 1, episode_rewards[-1]))
    plot_durations(episode_rewards)

    # average = sum(render_times_av) / len(render_times_av)
    # print("    av time to render =", average)
    # render_times.append(average)

    average = sum(action_times_av) / len(action_times_av)
    print("    av time to pick action =", average)
    action_times.append(average)

    average = sum(environment_times_av) / len(environment_times_av)
    print("    av time to take action in env =", average)
    environment_times.append(average)

    average = sum(train_times_av) / len(train_times_av)
    print("    av time to train network =", average)
    train_times.append(average)

if training:
    with open("params/episode_rewards.pkl", "wb") as f:
        pickle.dump(episode_rewards, f)

    torch.save(agent.policy_network.state_dict(), "params/policy_network.pt")
    torch.save(agent.target_network.state_dict(), "params/target_network.pt")

    with open("params/buffer.pkl", "wb") as f:
        pickle.dump(agent.memory.buffer, f)

env.close()

print("Total time averages:")

# average = sum(render_times) / len(render_times)
# print("\n    av time to render =", average)

average = sum(action_times) / len(action_times)
print("    \nav time to pick action =", average)

average = sum(environment_times) / len(environment_times)
print("    av time to take action in env =", average)

average = sum(train_times) / len(train_times)
print("    av time to train network =", average)
