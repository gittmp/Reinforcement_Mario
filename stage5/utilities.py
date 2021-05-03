import matplotlib.pyplot as plt
import os


# UTILITIES FOR ENVIRONMENT.PY
# action space reductions
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


# UTILITIES FOR AGENT.PY
# plot function to visualise downsampled state
def render_state(four_frames):
    single_image = four_frames.squeeze(0)[-1]
    fig = plt.figure("Frame")
    plt.imshow(single_image, cmap='gray')
    plt.title("Down-sampled image")
    plt.draw()
    plt.pause(0.001)


# make sure all files exist if we are loading pretrained network
def check_files(path):
    b = os.path.isfile(path + "policy_network.pt")
    b = b and os.path.isfile(path + "target_network.pt")
    b = b and os.path.isfile(path + "extrinsic_rewards.pkl")

    return b


# UTILITIES FOR RUN.PY
# print arguments which have been configured by user to log file
def print_args(dest, arg_dict):
    with open(dest + 'log.out', 'w') as f:
        for item in arg_dict:
            parameter = str(item) + ': ' + str(arg_dict[item])
            print(parameter)
            f.write(parameter + '\n')

        f.write("\nStarting episodes...\n")
