import matplotlib.pyplot as plt
import torch
import os


# utilities for environment.py
# Limiting action set combinations:
# actions for the simple run right environment
# right + B = run; hold A = jump higher; right + A + B = running jump;
# B = throw fire; water A = bob; down = crouch; start = play/pause;
ACTION_SET = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['down']
]

ACTION_SET2 = [
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
    ['down']
]


# utilities for agent.py
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
    plt.title("Down-sampled image")
    plt.draw()
    plt.pause(0.001)


def check_files(path):
    b = os.path.isfile(path + "policy_network.pt")
    b = b and os.path.isfile(path + "target_network.pt")
    b = b and os.path.isfile(path + "extrinsic_rewards.pkl")

    return b


# utilities for run.py
def print_args(dest, arg_dict):
    with open(dest + 'log.out', 'w') as f:
        for item in arg_dict:
            parameter = str(item) + ': ' + str(arg_dict[item])
            print(parameter)
            f.write(parameter + '\n')

        f.write("\nStarting episodes...\n")
