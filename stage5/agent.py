import math
from tqdm import tqdm
import time
import pickle

from network import *
from memory import *
from utilities import *


class Agent:
    def __init__(self, state_shape, action_n,
                 alpha, gamma, epsilon_ceil, epsilon_floor, epsilon_decay,
                 buffer_capacity, batch_size, update_target, path, episodes,
                 pretrained=False, plot=False, training=False,
                 network=1, memory=2, env_version=2):

        # configurable parameters of agent
        self.n_eps = episodes
        self.path = path
        self.mem_version = memory
        self.env_version = env_version
        self.training_times = []

        self.state_shape = state_shape
        self.action_n = action_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_ceil
        self.epsilon_ceil = epsilon_ceil
        self.epsilon_floor = epsilon_floor
        self.epsilon_decay = epsilon_decay
        self.update_target = update_target
        self.pretrained = pretrained and check_files(self.path)
        self.batch_size = batch_size
        self.plot = plot
        self.training = training
        self.train_step = 0

        # for when using NCC (so we can exploit GPU processing)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # loss function for backpropagation is the smoothl1loss
        if self.mem_version == 2:
            self.loss = nn.SmoothL1Loss(reduction='none').to(self.device)
        else:  # self.version == 0 or 1
            self.loss = nn.SmoothL1Loss().to(self.device)

        # network architecture
        if network == 0:
            self.policy_network = Network0(state_shape, action_n).to(self.device)
            self.target_network = Network0(state_shape, action_n).to(self.device)
        else:  # network == 1
            self.policy_network = Network1(state_shape, action_n).to(self.device)
            self.target_network = Network1(state_shape, action_n).to(self.device)

        # load parameters into network / data arrays if pretrained
        if self.pretrained:
            self.policy_network.load_state_dict(torch.load(self.path + "policy_network.pt", map_location=torch.device(self.device)))
            self.target_network.load_state_dict(torch.load(self.path + "target_network.pt", map_location=torch.device(self.device)))

            with open(self.path + 'log.out', 'a') as f:
                f.write("\nLoaded policy network from path = {} \n".format(self.path + "policy_network.pt"))
                f.write("Loaded target network from path = {} \n".format(self.path + "target_network.pt"))

            with open(self.path + "extrinsic_rewards.pkl", "rb") as f:
                self.extrinsic_rewards = pickle.load(f)

            if env_version == 2:
                with open(self.path + "intrinsic_rewards.pkl", "rb") as f:
                    self.intrinsic_rewards = pickle.load(f)

            with open(self.path + "distance_array.pkl", "rb") as f:
                self.distance_array = pickle.load(f)

            with open(self.path + 'log.out', 'a') as f:
                f.write("Loaded rewards over {} episodes from path = {} \n".format(len(self.extrinsic_rewards), self.path))
        else:
            self.extrinsic_rewards = []
            self.distance_array = []

            if env_version == 2:
                self.intrinsic_rewards = []

        # use Adam optimisation
        self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=alpha)

        # memory system
        if self.mem_version == 2:
            self.memory = PrioritisedMemory(100000, self.state_shape, self.device)
        elif self.mem_version == 1:
            self.memory = BasicMemory(self.state_shape, buffer_capacity, self.batch_size, self.pretrained, self.device, self.path, self.n_eps)

    def step(self, state):
        # conduct epsilon-greedy exploration
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_floor)

        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.action_n)]])
        else:
            nn_out = self.policy_network(state.to(self.device))
            return torch.argmax(nn_out).unsqueeze(0).unsqueeze(0).cpu()

    def target_update(self):
        # update target network to latest policy network parameters
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def train(self, exp):
        self.train_step += 1

        # push + sample batch of experiences from memory system (depending on memory configuration)
        if self.mem_version == 2:
            # calculate error associated with latest experience, and update prioritised memory with it
            S = torch.zeros(1, *self.state_shape).to(self.device)
            A = torch.zeros(1, 1).to(self.device)
            R = torch.zeros(1, 1).to(self.device)
            Succ = torch.zeros(1, *self.state_shape).to(self.device)
            T = torch.zeros(1, 1).to(self.device)
            S[0], A[0], R[0], Succ[0], T[0] = exp

            target = (R + torch.mul((self.gamma * self.target_network(Succ).max(1).values.unsqueeze(1)), 1 - T)).to(self.device)
            q_val = self.policy_network(S).gather(1, A.long()).to(self.device)
            td_error = torch.abs(target - q_val)
            self.memory.push(td_error, exp)

            # if memory buffer not full enough, stop here
            if not self.memory.full():
                return

            # otherwise, sample batch and corresponding weighting
            batch, indices, weights = self.memory.sample(self.batch_size)

            if batch is None:
                return

            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards']
            successors = batch['successors']
            terminals = batch['terminals']

        elif self.mem_version == 1:
            # update basic memory buffer
            self.memory.push(exp)

            # sample batch if memory full enough
            if not self.memory.full():
                return

            batch = self.memory.sample()
            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards']
            successors = batch['successors']
            terminals = batch['terminals']

        else:  # self.mem_version == 0
            # for baseline memory system, set prepare experience for network
            states = torch.zeros(1, *self.state_shape).to(self.device)
            actions = torch.zeros(1, 1).to(self.device)
            rewards = torch.zeros(1, 1).to(self.device)
            successors = torch.zeros(1, *self.state_shape).to(self.device)
            terminals = torch.zeros(1, 1).to(self.device)

            states[0], actions[0], rewards[0], successors[0], terminals[0] = exp

        # use backpropagation to calculate the gradient of the loss function
        # Q-target = reward + discount factor * successor q-value
        targets = (rewards + torch.mul((self.gamma * self.target_network(successors).max(1).values.unsqueeze(1)), 1 - terminals)).to(self.device)
        q_vals = self.policy_network(states).gather(1, actions.long()).to(self.device)

        if self.mem_version == 2:
            # update prioritised memory with new error terms (error = q-target - q-value)
            errors = torch.abs(targets - q_vals)
            self.memory.update(indices, errors)

            # calculate weighted loss function
            loss = self.loss(q_vals, targets)
            loss = torch.mul(loss, weights).mean()
        else:  # self.version == 0 or 1
            loss = self.loss(q_vals, targets)

        # backpropagation + gradient descent
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # update target network every self.update_target steps
        if self.train_step % self.update_target == 0:
            self.target_update()

    # save parameters and data
    def save(self):
        torch.save(self.policy_network.state_dict(), self.path + "policy_network.pt")
        torch.save(self.target_network.state_dict(), self.path + "target_network.pt")

        with open(self.path + "extrinsic_rewards.pkl", "wb") as f:
            pickle.dump(self.extrinsic_rewards, f)

        if self.env_version == 2:
            with open(self.path + "intrinsic_rewards.pkl", "wb") as f:
                pickle.dump(self.intrinsic_rewards, f)

        with open(self.path + "distance_array.pkl", "wb") as f:
            pickle.dump(self.distance_array, f)

    # run the agent
    def run(self, env, eps):

        for _ in tqdm(range(eps)):
            # start state
            state = env.reset()
            state = torch.Tensor([state])
            timestep = 0
            total_reward = 0

            while True:
                timestep += 1

                if self.plot:
                    env.render()

                    # if timestep % 10 == 0:
                    #     render_state(state)

                # select action from agent's policy
                action = self.step(state)

                # employ action, to receive transition of experience
                successor, reward, terminal, info = env.step(int(action[0]))
                successor = torch.Tensor([successor])
                total_reward += reward

                # conduct training process
                if self.training:
                    start = time.time()

                    experience = (
                        state.float(),
                        action.float(),
                        torch.Tensor([reward]).unsqueeze(0).float(),
                        successor.float(),
                        torch.Tensor([int(terminal)]).unsqueeze(0).float()
                    )
                    self.train(experience)

                    self.training_times.append(time.time() - start)

                # move to next state
                state = successor

                # collect data if episode has terminated
                if terminal:
                    # print(f"Distance covered = {info['x_pos']}")
                    if self.env_version == 2:
                        self.intrinsic_rewards.append(total_reward)

                    self.extrinsic_rewards.append(info['score'])
                    self.distance_array.append(info['x_pos'])
                    break

        # save network parameters and data at end of training process
        if self.training:
            with open(self.path + 'log.out', 'a') as f:
                f.write("\nSaving final parameters! \n")
            self.save()

    # print data collected over the run
    def print_stats(self, no_plot_points=25):

        with open(self.path + 'log.out', 'a') as f:
            f.write("\nTotal episodes trained over: {} \n".format(len(self.extrinsic_rewards)))

            sections = min(no_plot_points, self.n_eps)
            section_size = math.floor(len(self.extrinsic_rewards) / sections)

            # print table of extrinsic rewards (game scores) over session
            f.write("\n\nAverage extrinsic rewards over past {} episodes: \n".format(len(self.extrinsic_rewards)))
            f.write("EPISODE RANGE                AV. EXTRINSIC REWARD \n")

            for i in range(sections):
                low = i * section_size
                high = (i + 1) * section_size

                if i == sections - 1:
                    av = sum(self.extrinsic_rewards[low:]) / (len(self.extrinsic_rewards) - low)
                    f.write("[{}, {}) {} {} \n".format(low, len(self.extrinsic_rewards), " " * (25 - 2 - len(str(low)) - len(str(len(self.extrinsic_rewards)))), av))
                else:
                    av = sum(self.extrinsic_rewards[low:high]) / (high - low)
                    f.write("[{}, {}) {} {} \n".format(low, high, " " * (25 - 2 - len(str(low)) - len(str(high))), av))

            if self.env_version == 2:
                # print table of intrinsic rewards (manufactured reward signal) over session
                f.write("\n\nAverage intrinsic rewards over past {} episodes: \n".format(len(self.intrinsic_rewards)))
                f.write("EPISODE RANGE                AV. INTRINSIC REWARD \n")
                section_size = math.floor(len(self.intrinsic_rewards) / sections)

                for i in range(sections):
                    low = i * section_size
                    high = (i + 1) * section_size

                    if i == sections - 1:
                        av = sum(self.intrinsic_rewards[low:]) / (len(self.intrinsic_rewards) - low)
                        f.write("[{}, {}) {} {} \n".format(low, len(self.intrinsic_rewards), " " * (25 - 2 - len(str(low)) - len(str(len(self.intrinsic_rewards)))), av))
                    else:
                        av = sum(self.intrinsic_rewards[low:high]) / (high - low)
                        f.write("[{}, {}) {} {} \n".format(low, high, " " * (25 - 2 - len(str(low)) - len(str(high))), av))

            # print table of x distance walked over session
            f.write("\n\nAverage x distance travelled over past {} episodes: \n".format(len(self.distance_array)))
            f.write("EPISODE RANGE                AV. DISTANCE \n")
            section_size = math.floor(len(self.distance_array) / sections)

            for i in range(sections):
                low = i * section_size
                high = (i + 1) * section_size

                if i == sections - 1:
                    av = sum(self.distance_array[low:]) / (len(self.distance_array) - low)
                    f.write("[{}, {}) {} {} \n".format(low, len(self.distance_array), " " * (25 - 2 - len(str(low)) - len(str(len(self.distance_array)))), av))
                else:
                    av = sum(self.distance_array[low:high]) / (high - low)
                    f.write("[{}, {}) {} {} \n".format(low, high, " " * (25 - 2 - len(str(low)) - len(str(high))), av))

            # print training times for each timestep over session
            f.write("\n\nAverage training time per time-step over past {} time-steps: {}".format(len(self.training_times), sum(self.training_times) / len(self.distance_array)))
