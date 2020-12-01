import torch
import gym_super_mario_bros as gym_smb
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt

import processed_mario_wrappers as wrappers
import mario_network as network


game = 'SuperMarioBros-1-1-v0'
env = gym_smb.make(game)
env = wrappers.make_env(env)

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
    pretrained=False
)

training = True
no_eps = 100
env.reset()
total_rewards = []

for ep in tqdm(range(no_eps)):
    state = env.reset()
    state = torch.Tensor([state])
    total_reward = 0
    timestep = 0

    while True:
        timestep += 1
        env.render()

        action = agent.step(state)
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

    total_rewards.append(total_reward)
    print("\nTotal reward after episode {} is {}".format(ep + 1, total_rewards[-1]))

if training:
    with open("total_rewards.pkl", "wb") as f:
        pickle.dump(total_rewards, f)

    torch.save(agent.policy_network.state_dict(), "policy_network.pt")
    torch.save(agent.target_network.state_dict(), "target_network.pt")

    with open("buffer.pkl", "wb") as f:
        pickle.dump(agent.memory.buffer, f)

env.close()

plt.title("Episodes trained vs. Average Rewards (per 500 eps)")
plt.plot([0 for _ in range(500)] + np.convolve(total_rewards, np.ones((500,))/500, mode="valid").tolist())
plt.show()
