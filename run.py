import argparse
import stage4.agent2.main as agent2
import stage4.agent3.main as agent3
import stage4.agent4.main as agent4


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reinforcement Mario')
    parser.add_argument('--A', dest='agent', type=int, help='Agent version to utilise: [2, 3, 4]', required=True)
    parser.add_argument('--N', dest='no_eps', type=int, help='The number of episodes to run through', required=True)
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

    agent = args.agent
    no_eps = args.no_eps
    world = args.world
    training = args.training
    plot = args.plot
    pretrained = args.pretrained
    ncc = args.ncc

    if agent == 2:
        path = 'stage4/agent2/'

        if ncc:
            path += 'ncc_params2/'
        else:
            path += 'params2/'

        with open(path + f'log4-{no_eps}.out', 'a') as f:
            f.write("\nAGENT {} SELECTED! \nStarting episodes...\n".format(agent))

        agent2.run(no_eps=no_eps, training=training, pretrained=pretrained, plot=plot, world=world, path=path)

    elif agent == 3:
        path = 'stage4/agent3/'

        if ncc:
            path += 'ncc_params3/'
        else:
            path += 'params3/'

        with open(path + f'log4-{no_eps}.out', 'a') as f:
            f.write("\nAGENT {} SELECTED! \nStarting episodes...\n".format(agent))

        agent3.run(no_eps=no_eps, training=training, pretrained=pretrained, plot=plot, world=world, path=path)

    elif agent == 4:
        path = 'stage4/agent4/'

        if ncc:
            path += 'ncc_params4/'
        else:
            path += 'params4/'

        with open(path + f'log4-{no_eps}.out', 'a') as f:
            f.write("\nAGENT {} SELECTED! \nStarting episodes...\n".format(agent))

        agent4.run(no_eps=no_eps, training=training, pretrained=pretrained, plot=plot, world=world, path=path)

    else:
        print("ERROR: invalid agent selected!")
        exit(1)
