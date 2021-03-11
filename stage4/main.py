import os
import math
import torch
from tqdm import tqdm
import pickle

from network import Agent
from environment import make_env, plot_durations, render_state

game = 'SuperMarioBros-1-1-v0'
env = make_env(game)
training = True
ncc = False
plot = False
pretrained = False
no_eps = 25

if ncc:
    path = "ncc_params4/"
else:
    path = "params4/"

pretrained = pretrained and os.path.isfile(path + "policy_network.pt")

if pretrained:
    with open(path + "episode_rewards.pkl", "rb") as f:
        episode_rewards = pickle.load(f)
        print("Loaded rewards over {} episodes from path = {}".format(len(episode_rewards), path + "episode_rewards.pkl"))
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
    pretrained=pretrained
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

        if training:
            experience = (
                state.float(),
                action.float(),
                torch.Tensor([reward]).unsqueeze(0).float(),
                successor.float(),
                torch.Tensor([int(terminal)]).unsqueeze(0).float()
            )
            agent.train(experience)

        state = successor

        if terminal:
            # print("\nInfo:\nfinal game score = {}, time elapsed = {}, Mario's location = ({}, {})"
            #       .format(info['score'], 400 - info['time'], info['x_pos'], info['y_pos']))

            # if plot:
            #     plot_durations(episode_rewards)

            break

    if training:
        # episode_rewards.append(total_reward)
        episode_rewards.append(info['score'])
        print("Game score after termination = {}".format(info['score']))

        if ep % max(1.0, math.floor(no_eps / 4)) == 0:
            print("automatically saving params4 at episode {}".format(ep))

            with open(path + "episode_rewards.pkl", "wb") as f:
                pickle.dump(episode_rewards, f)

            # with open(path + "buffer.pkl", "wb") as f:
            #     pickle.dump(agent.memory.tree, f)

            torch.save(agent.policy_network.state_dict(), path + "policy_network.pt")
            torch.save(agent.target_network.state_dict(), path + "target_network.pt")

if training:
    with open(path + "episode_rewards.pkl", "wb") as f:
        pickle.dump(episode_rewards, f)

    # with open(path + "buffer.pkl", "wb") as f:
    #     pickle.dump(agent.memory.tree, f)

    torch.save(agent.policy_network.state_dict(), path + "policy_network.pt")
    torch.save(agent.target_network.state_dict(), path + "target_network.pt")

    print("Final parameters saved!")

env.close()

print("\nTRAINING COMPLETE")
print("Total episodes trained over:", len(episode_rewards))
print("Average reward over all episodes:", sum(episode_rewards)/len(episode_rewards))
print("Average reward over last training session:", sum(episode_rewards[-no_eps:])/no_eps)
