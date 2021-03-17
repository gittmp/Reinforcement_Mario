import argparse
from agent4.network import *
from agent4.environment import *


def run(no_eps=10000, training=True, pretrained=False, plot=False, world=1, path=None):

    if world == 2:
        game = 'SuperMarioBros2-v0'
    else:
        game = 'SuperMarioBros-v0'

    with open(path + f'log4-{no_eps}.out', 'w') as f:
        f.write("Parameters:\n     no_eps={}, \n     world={}, game={}, \n     training={}, plot={}, \n     pretrained={}, path={} \n".format(no_eps, world, game, training, plot, pretrained, path))

    src = {
        'path': path,
        'eps': no_eps
    }

    env = make_env(game, src)

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
        eps=no_eps,
        pretrained=pretrained,
        path=path,
        plot=plot,
        training=training
    )

    with open(path + f'log4-{no_eps}.out', 'a') as f:
        f.write("\nStarting episodes...\n")

    env.reset()
    agent.run(env, no_eps)
    env.close()

    with open(path + f'log4-{no_eps}.out', 'a') as f:
        f.write("\nTraining complete!\n")

    agent.print_stats()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reinforcement Mario')
    parser.add_argument('--eps', dest='no_eps', type=int, help='The number of episodes to run through', required=True)
    parser.add_argument('--w', dest='world', type=int, help='The SMB world we wish to explore: [1, 2]', default=1)
    parser.add_argument('-t', dest='training', action='store_true',
                        help='True = train agent over episodes played; False = just run without training',
                        default=False)
    parser.add_argument('-plot', dest='plot', action='store_true',
                        help='True = plot total reward after each completed episode', default=False)
    parser.add_argument('-ptd', dest='pretrained', action='store_true',
                        help='True = agent has been pretrained with parameter files available; False = start agent from scratch',
                        default=False)
    parser.add_argument('-ncc', dest='ncc', action='store_true',
                        help='True = load pretrained data from NCC parameters; False = load from local parameters',
                        default=False)
    args = parser.parse_args()

    ne = args.no_eps
    w = args.world
    t = args.training
    p = args.plot
    ptd = args.pretrained
    ncc = args.ncc

    if ncc:
        path = '../../ncc_tests/ncc_params4/'
    else:
        path = 'params4/'

    run(no_eps=ne, training=t, pretrained=ptd, plot=p, world=w, path=path)
