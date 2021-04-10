import os
import math
import torch
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from agent3.network import *
from agent3.environment import *


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


def run(no_eps=10000, training=True, pretrained=False, plot=False, world=1, stage=1, version=0, path=None):

    game = 'SuperMarioBros' + str(world) + '-' + str(stage) + '-v' + str(version)

    src = {
        'path': path,
        'eps': no_eps
    }

    env = make_env(game, src)

    pretrained = pretrained and os.path.isfile(path + "policy_network.pt")

    if pretrained:
        with open(path + "episode_rewards.pkl", "rb") as f:
            episode_rewards = pickle.load(f)
            print("Loaded rewards over {} episodes from path = {}".format(len(episode_rewards), path + 'episode_rewards.pkl'))
    else:
        episode_rewards = []

    env.reset()

    agent = Agent(
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
        pretrained=pretrained,
        source=src
    )

    print("\nStarting episodes...\n")
    for ep in tqdm(range(no_eps)):
        state = env.reset()
        state = torch.Tensor([state])
        # total_reward = 0
        timestep = 0

        while True:
            timestep += 1

            if plot:
                env.render()

                # if timestep % 10 == 0:
                #     render_state(state)

            action = agent.step(state)

            successor, reward, terminal, info = env.step(int(action[0]))
            # total_reward += reward

            successor = torch.Tensor([successor])
            reward = torch.Tensor([reward]).unsqueeze(0)
            terminal = torch.Tensor([int(terminal)]).unsqueeze(0)

            if training:
                experience = (state.float(), action.float(), reward.float(), successor.float(), terminal.float())
                agent.memory.push(experience)
                agent.train()

            state = successor

            if terminal:
                # print("\nInfo:\nfinal game score = {}, time elapsed = {}, Mario's location = ({}, {})"
                #       .format(info['score'], 400 - info['time'], info['x_pos'], info['y_pos']))

                if plot:
                    plot_durations(episode_rewards)
                break

        if training:
            # episode_rewards.append(total_reward)
            episode_rewards.append(info['score'])
            print("\nGame score after termination = {}".format(info['score']))

            if plot:
                plot_durations(episode_rewards)

            if ep % math.floor(no_eps / 4) == 0:
                print("automatically saving prams at episode {}".format(ep))

                with open(path + "episode_rewards.pkl", "wb") as f:
                    pickle.dump(episode_rewards, f)

                with open(path + "buffer.pkl", "wb") as f:
                    pickle.dump(agent.memory.buffer, f)

                torch.save(agent.policy_network.state_dict(), path + "policy_network.pt")
                torch.save(agent.target_network.state_dict(), path + "target_network.pt")

    if training:
        with open(path + "episode_rewards.pkl", "wb") as f:
            pickle.dump(episode_rewards, f)

        with open(path + "buffer.pkl", "wb") as f:
            pickle.dump(agent.memory.buffer, f)

        torch.save(agent.policy_network.state_dict(), path + "policy_network.pt")
        torch.save(agent.target_network.state_dict(), path + "target_network.pt")

        print("Final parameters saved!")

    env.close()

    print("\nTRAINING COMPLETE")
    print("Total episodes trained over:", len(episode_rewards))
    print("Average reward over all episodes:", sum(episode_rewards)/len(episode_rewards))
    print("Average reward over last training session:", sum(episode_rewards[-no_eps:])/no_eps)
