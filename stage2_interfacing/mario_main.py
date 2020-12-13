import torch
import gym_super_mario_bros as gym_smb
from tqdm import tqdm
# import numpy as np
import pickle
import matplotlib.pyplot as plt

import mario_wrapper as wrapper
import mario_network as network

# import retro


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

pretrained = False

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
    with open("episode_rewards.pkl", "rb") as f:
        episode_rewards = pickle.load(f)
else:
    episode_rewards = []

training = True
no_eps = 100
env.reset()

for ep in tqdm(range(no_eps)):
    state = env.reset()
    state = torch.Tensor([state])
    total_reward = 0
    timestep = 0

    while True:
        timestep += 1
        # env.render()

        action = agent.step(state)

        # sample_action = torch.Tensor(env.action_space.sample())
        # print("Chosen action = ", action)
        # print("Sampled action = ", sample_action)

        successor, reward, terminal, info = env.step(int(action[0]))
        total_reward += reward

        successor = torch.Tensor([successor])
        reward = torch.Tensor([reward]).unsqueeze(0)
        terminal = torch.Tensor([int(terminal)]).unsqueeze(0)

        if training:
            experience = (state.float(), action.float(), reward.float(), successor.float(), terminal.float())
            agent.memory.push(experience)
            agent.train()

        state = successor

        if terminal:
            print("Info:\nfinal game score = {}, time elapsed = {}, Mario's location = ({}, {})"
                  .format(info['score'], 400 - info['time'], info['x_pos'], info['y_pos']))
            break

    episode_rewards.append(total_reward)
    print("\nTotal reward after episode {} is {}".format(ep + 1, episode_rewards[-1]))
    plot_durations(episode_rewards)

if training:
    with open("episode_rewards.pkl", "wb") as f:
        pickle.dump(episode_rewards, f)

    torch.save(agent.policy_network.state_dict(), "policy_network.pt")
    torch.save(agent.target_network.state_dict(), "target_network.pt")

    with open("buffer.pkl", "wb") as f:
        pickle.dump(agent.memory.buffer, f)

env.close()
