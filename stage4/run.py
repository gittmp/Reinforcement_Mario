import argparse
import agent2.main as agent2
import agent3.main as agent3
import agent4.main as agent4


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reinforcement Mario')
    parser.add_argument('--A', dest='agent', type=int, help='Agent version to utilise: {2, 3, 4}', required=True)
    parser.add_argument('--N', dest='no_eps', type=int, help='The number of episodes to run through', required=True)
    parser.add_argument('--nwk', dest='network', type=int, help='Network version to utilise: {0, 1}', default=1)
    parser.add_argument('--w', dest='world', type=int, help='The SMB world we wish to explore: {1, 2, 3, 4, 5, 6, 7, 8}', default=1)
    parser.add_argument('--s', dest='stage', type=int, help='The stage of the SMB world we wish to explore: {1, 2, 3, 4}', default=1)
    parser.add_argument('--v', dest='version', type=int, help='The ROM version of SMB we wish to explore: {0, 1, 2, 3}', default=0)
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

    agent = args.agent
    no_eps = args.no_eps
    world = args.world
    stage = args.stage
    version = args.version
    training = args.training
    plot = args.plot
    pretrained = args.pretrained
    ncc = args.ncc
    network = args.network

    if agent == 2:

        if ncc:
            path = '../stage5/ncc_tests/ncc_params2/'
        else:
            path = 'agent2/params2/'

        with open(path + f'log4-{no_eps}.out', 'w') as f:
            f.write("\nAGENT {} SELECTED! \nStarting episodes...\n".format(agent))

        agent2.run(no_eps=no_eps, training=training, pretrained=pretrained, plot=plot, world=world, stage=stage, version=version, path=path)

    elif agent == 3:

        if ncc:
            path = '../stage5/ncc_tests/ncc_params3/'
        else:
            path = 'agent3/params3/'

        with open(path + f'log4-{no_eps}.out', 'w') as f:
            f.write("\nAGENT {} SELECTED! \nStarting episodes...\n".format(agent))

        agent3.run(no_eps=no_eps, training=training, pretrained=pretrained, plot=plot, world=world, stage=stage, version=version, path=path)

    elif agent == 4:

        if ncc:
            path = '../stage5/ncc_tests/ncc_params4/'
        else:
            path = 'agent4/params4/'

        with open(path + f'log4-{no_eps}.out', 'w') as f:
            f.write("\nAGENT {} SELECTED! \nStarting episodes...\n".format(agent))

        agent4.run(no_eps=no_eps, training=training, pretrained=pretrained, plot=plot, world=world, stage=stage, version=version, path=path, net=network)

    else:
        print("ERROR: invalid agent selected!")
        exit(1)
