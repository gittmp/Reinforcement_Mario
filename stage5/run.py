import argparse
import os
from main import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reinforcement Mario')
    parser.add_argument('--N', dest='no_eps', type=int, help='The number of episodes to run through', required=True)
    parser.add_argument('--nwk', dest='network', type=int, help='Network version to utilise: {0, 1}', default=1)
    parser.add_argument('--m', dest='memory', type=int, help='Memory implemnetation to use: {0 = Basic, 1 = PER}', default=1)
    parser.add_argument('--w', dest='world', type=int, help='The SMB world we wish to explore: {1, 2, 3, 4, 5, 6, 7, 8}', default=1)
    parser.add_argument('--s', dest='stage', type=int, help='The stage of the SMB world we wish to explore: {1, 2, 3, 4}', default=1)
    parser.add_argument('--v', dest='rom', type=int, help='The ROM version of SMB we wish to explore: {0, 1, 2, 3}', default=0)
    parser.add_argument('--env', dest='env_version', type=int, help='Environment manipulation version to utilise: {0 = None, 1 = State & action, 2 = Reward}', default=2)
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

    no_eps = args.no_eps
    memory = args.memory
    world = args.world
    stage = args.stage
    rom = args.rom
    env_version = args.env_version
    training = args.training
    plot = args.plot
    pretrained = args.pretrained
    ncc = args.ncc
    network = args.network

    print("args:", args)

    if ncc:
        path = 'ncc_tests/'
        dir_count = len(list(filter(os.path.isdir, os.listdir(path))))
        path = os.path.join(path, 'test' + str(dir_count))
        os.mkdir(path)
    else:
        path = 'params5/'

    with open(path + f'log4-{no_eps}.out', 'w') as f:
        f.write("\nStarting episodes...\n")

    run(no_eps=no_eps, training=training, pretrained=pretrained, plot=plot, world=world, stage=stage, version=rom, path=path, net=network, mem=memory, env_vers=env_version)

