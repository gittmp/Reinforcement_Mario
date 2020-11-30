# based upon: https://ml-showcase.paperspace.com/projects/super-mario-bros-double-deep-q-network
import torch
import gym_super_mario_bros  # SMB environment integrated into OpenAI Gym
from tqdm import tqdm  # progress bar
import pickle  # util for converting between python object structures and byte streams
import numpy as np
import matplotlib.pyplot as plt

import processed_mario_dqn as dqn
import processed_mario_wrappers as wrappers


def run(training_mode, pretrained):
    # create environment and wrap so that frames are downscaled and grayscale
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = wrappers.make_env(env)

    # get size of state and action spaces
    observation_space = env.observation_space.shape
    action_space = env.action_space.n

    # create agent which acts in the environment
    agent = dqn.Agent(state_space=observation_space,
                      action_space=action_space,
                      max_memory_size=30000,
                      batch_size=32,
                      gamma=0.90,
                      lr=0.00025,
                      exploration_max=1.0,
                      exploration_min=0.02,
                      exploration_decay=0.99,
                      pretrained=pretrained)

    # run through episodes
    num_episodes = 10000
    env.reset()
    total_rewards = []

    # tqdm shows progress bar for no. eps completed - abbreviated from Spanish phrase for "I love you so much"?????
    for ep in tqdm(range(num_episodes)):
        # ep initialisation
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0

        # episode time-steps
        while True:
            steps += 1
            env.render()

            # generate action from the agent's policy network
            action = agent.act(state)

            # get observations from environment based upon the action
            state_next, reward, terminal, info = env.step(int(action[0]))
            total_reward += reward

            # represent successor state, reward and terminal bool as tensors
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            # if training: add experience tuple, and conduct experience replay training
            if training_mode:
                agent.remember(state, action, reward, state_next, terminal)
                agent.experience_replay()

            # progress to successor state
            state = state_next

            # break if end of the episode
            if terminal:
                print("Info:\nfinal game score = {}, time elapsed = {}, Mario's location = ({}, {})"
                      .format(info['score'], 400 - info['time'], info['x_pos'], info['y_pos']))
                break

        # evaluation metrics
        total_rewards.append(total_reward)
        print("\nTotal reward after episode {} is {}".format(ep + 1, total_rewards[-1]))
        num_episodes += 1

    # if training: deposit learned model into files
    if training_mode:
        with open("ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open("num_in_queue.pkl", "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open("total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)

        torch.save(agent.local_net.state_dict(), "dq1.pt")
        torch.save(agent.target_net.state_dict(), "dq2.pt")

        torch.save(agent.STATE_MEM, "STATE_MEM.pt")
        torch.save(agent.ACTION_MEM, "ACTION_MEM.pt")
        torch.save(agent.REWARD_MEM, "REWARD_MEM.pt")
        torch.save(agent.STATE2_MEM, "STATE2_MEM.pt")
        torch.save(agent.DONE_MEM, "DONE_MEM.pt")

    env.close()

    if num_episodes > 500:
        plt.title("Episodes trained vs. Average Rewards (per 500 eps)")
        plt.plot([0 for _ in range(500)] +
                 np.convolve(total_rewards, np.ones((500,))/500, mode="valid").tolist())
        plt.show()


run(training_mode=True, pretrained=False)
