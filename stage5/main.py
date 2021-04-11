import argparse
from network import *
from environment import *


def run(no_eps=10000, training=True, pretrained=False, plot=False, world=1, stage=1, version=0, path=None, net=1, mem=1, env_vers=2):

    game = 'SuperMarioBros-' + str(world) + '-' + str(stage) + '-v' + str(version)

    with open(path + f'log-{no_eps}.out', 'w') as f:
        f.write("Parameters:\n     no_eps={}, \n     world={}, game={}, \n     training={}, plot={}, \n     pretrained={}, path={} \n".format(no_eps, world, game, training, plot, pretrained, path))

    src = {
        'path': path,
        'eps': no_eps
    }

    env = make_env(game, src, env_vers)

    agent = Agent(
        state_shape=env.observation_space.shape,
        action_n=env.action_space.n,
        alpha=0.00025,
        gamma=0.9,
        epsilon_ceil=1.0,
        epsilon_floor=0.02,
        epsilon_decay=0.99,
        buffer_capacity=30000,
        batch_size=64,
        update_target=5000,
        source=src,
        pretrained=pretrained,
        plot=plot,
        training=training,
        network=net,
        memory=mem
    )

    with open(path + f'log-{no_eps}.out', 'a') as f:
        f.write("\nStarting episodes...\n")

    env.reset()
    agent.run(env, no_eps)
    env.close()

    with open(path + f'log-{no_eps}.out', 'a') as f:
        f.write("\nTraining complete!\n")

    agent.print_stats()
