import numpy as np
import tensorflow as tf
import collections


# create variable for weights of NN layer
def shape_weights(shape, initialiser):
    return tf.Variable(
        initialiser(shape),
        trainable=True,
        dtype=tf.float32)


# calculates output of NN layer on input of previous layer (given corresponding weights/biases)
def calc_layer(x, weights, bias, activation=tf.identity, **activation_kwargs):
    z = tf.matmul(x, weights) + bias
    return activation(z, **activation_kwargs)


# simple neural network class for use as a Q-function approximator
class Network(object):
    # initialising params of NN
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size=None,
                 weights_initialiser=tf.initializers.glorot_uniform(),
                 bias_initialiser=tf.initializers.zeros(),
                 optimiser=tf.optimizers.Adam,
                 **optimiser_kwargs):

        # set NN sizes
        self.input_size = input_size
        self.output_size = output_size

        # if shape of NN not specified, initialise with two hidden layers of width 50
        if hidden_size is None:
            hidden_size = [50, 50]
        self.hidden_size = hidden_size

        # seed randomness to keep it uniform
        np.random.seed(41)

        # initialise weights and optimise network
        self.initialise_weights(weights_initialiser, bias_initialiser)
        self.optimiser = optimiser(**optimiser_kwargs)

    # for randomly initialising weights of the network
    def initialise_weights(self, weights_initialiser, bias_initialiser):
        # weights for all connections between each layer
        weight_shapes = [
            # input layer -> 1st hidden layer
            [self.input_size, self.hidden_size[0]],
            # 1st hidden -> 2nd hidden
            [self.hidden_size[0], self.hidden_size[1]],
            # 2nd hidden -> output layer
            [self.hidden_size[1], self.output_size]
        ]

        # biases for each neuron in hidden layers + output layer
        bias_shapes = [
            # 1st hidden layer neurons
            [1, self.hidden_size[0]],
            # 2nd hidden layer neurons
            [1, self.hidden_size[1]],
            # output layer neurons
            [1, self.output_size]
        ]

        # set weights/biases of network to be initialised in correct shapes
        self.weights = [shape_weights(conn, weights_initialiser) for conn in weight_shapes]
        self.biases = [shape_weights(neuron, bias_initialiser) for neuron in bias_shapes]
        self.trainable_vars = self.weights + self.biases

    # NN model: input of state vectors batch, bringing together layers and outputting Q-values for each action per batch
    def model(self, inputs):
        # use ReLU function for scaling activations
        # 1st hidden layer takes input of NN inputs, 1st set of connection weights into it and biases for its neurons
        hidden1 = calc_layer(inputs, self.weights[0], self.biases[0], tf.nn.relu)
        hidden2 = calc_layer(hidden1, self.weights[1], self.biases[1], tf.nn.relu)
        # output layer activations aren't scaled (uses default identity function)
        output_layer = calc_layer(hidden2, self.weights[2], self.biases[2])

        return output_layer

    # function to complete one step of training, updating weights to minimise the distance between inputs and targets
    def train_step(self, inputs, targets, actions_one_hot):
        with tf.GradientTape() as tape:
            # get action value outputs of network on that input
            qvalues = tf.squeeze(self.model(inputs))
            # uses one-hot encoding (shows which action picked) to only show q-value of that picked action for each element in the batch
            onehot_qvalues = tf.reduce_sum(qvalues * actions_one_hot, axis=1)
            # calculate loss function as the MSE between target output and these masked one hot q values
            # print("targets: ", targets)
            mse_loss = tf.losses.mean_squared_error(targets, onehot_qvalues)

        # calculate the gradient along which we should update network weights to minimise the MSE loss
        gradients = tape.gradient(mse_loss, self.trainable_vars)

        # apply this gradient descent update to the network variables
        self.optimiser.apply_gradients(zip(gradients, self.trainable_vars))


# memory buffer for experience replay functionality
class Memory(object):
    # initialise memory buffer of experience tuples as a double ended queue of maximum length max_size
    def __init__(self):
        self.buffer = collections.deque(maxlen=50000)

    # function to return the length of the buffer
    def __len__(self):
        return len(self.buffer)

    # function to add experience tuple to memory buffer
    def add(self, experience):
        self.buffer = np.append(self.buffer, experience)

    # function to randomly sample a batch (of size batch_size) of experience tuples from the memory buffer
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        # print("Batch size = ", batch_size, " Buffer size = ", buffer_size)

        # choose a random collection of batch_size indices in range of buffer_size
        random_indices = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False
        )

        # find the corresponding experience tuples to these indices
        random_tuples = [self.buffer[i] for i in random_indices]

        return random_tuples


# our DQL agent which exploits the defined DQN to choose actions in the environment
class Agent(object):
    # initialise the params of the agent's Q-network
    def __init__(self,
                 state_input_size,
                 action_space_size,
                 target_update_freq=1000,
                 discount_factor=0.99,
                 batch_size=32,
                 epsilon_ceil=1,
                 epsilon_floor=0.05,
                 epsilon_anneal_rate=(1/100000),
                 ermb_init_size=10000):

        # utilise two NNs: the optimising network being continually updated and the fixed target network for behaviour
        self.action_space_size = action_space_size
        self.optimising_network = Network(state_input_size, action_space_size)
        self.target_network = Network(state_input_size, action_space_size)

        # initially update the fixed target network
        self.update_target_network()

        # set initial training parameters
        self.target_update_freq = target_update_freq
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.epsilon_ceil = epsilon_ceil + (epsilon_anneal_rate * ermb_init_size)
        self.epsilon_floor = epsilon_floor
        self.epsilon_anneal_rate = epsilon_anneal_rate
        self.timestep = 0

        # initialise the experience replay buffer
        self.memory = Memory()
        self.ermb_init_size = ermb_init_size

        # print('initialised agent')

    # handle the conditions at the start of an episode
    def start_episode(self):
        self.last_state = None
        self.last_action = None

    # function to progress one timestep (observes state/prev. reward, picks action, receives successor reward/state)
    def step(self, observation, training=True):
        # retrieve previous state-action pair from agent
        last_state = self.last_state
        last_action = self.last_action

        # retrieve the reward generated from the last state-action and the successor state we are now in
        last_reward = observation['reward']
        state = observation['state']

        # pick next action from observed state using behaviour policy
        # training bool indicates if we should use the target policy (during training) or optimised policy (afterwards)
        action = self.policy(state, training)
        # print("Picked action: ", action)

        if training:
            self.timestep += 1

            # if this isn't the initial state, add the experience tuple to the memory buffer
            if last_state is not None:
                experience = {
                    'state': last_state,
                    'action': last_action,
                    'reward': last_reward,
                    'successor': state
                }
                self.memory.add(experience)
                # print("Experience added to memory")

            # if the memory buffer is large enough to initiate experience replay, do training
            if self.timestep > self.ermb_init_size:
                # conduct experience replay training on optimising network
                # print("Training optimising network")
                self.train_network()

                # if target network at refresh threshold, update it
                if self.timestep % self.target_update_freq == 0:
                    # print("Training target network")
                    self.update_target_network()

            self.last_state = state
            self.last_action = action

        return action

    def policy(self, state, training):
        # policy to pick actions given current state
        # epsilon-greedy policy during training
        exploration_probability = self.epsilon_ceil - self.timestep * self.epsilon_anneal_rate

        if training and np.random.rand() < max(exploration_probability, self.epsilon_floor):
            # print("Random action chosen")
            action = np.random.randint(self.action_space_size)
        else:
            # afterwards, (or if not epsilon) just use greedy policy
            # print("Action chosen from policy")
            inputs = np.expand_dims(state, 0)
            qvalues = self.optimising_network.model(inputs)
            action = np.squeeze(np.argmax(qvalues, axis=-1))

        return action

    # function to unfix target network and update to current weights of optimising network, then fix it again there
    def update_target_network(self):
        optimised_vars = self.optimising_network.trainable_vars
        tf_vars = [tf.Variable(var) for var in optimised_vars]
        self.target_network.trainable_vars = tf_vars

    # function to update the weights of the optimising network minimising distance between inputs and targets
    def train_network(self):
        batch = self.memory.sample(self.batch_size)
        inputs = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        successor_inputs = np.array([exp['successor'] for exp in batch])

        # create one hot encoding vector of the action chosen from the network on corresponding input state from batch
        actions_one_hot = np.eye(self.action_space_size)[actions]

        # get the q values of the successor state by passing it through the target network used to pick actions
        successor_qvalues = np.squeeze(self.target_network.model(successor_inputs))

        # then use these successor q values to judge the worth of that successor state and use it in the q target
        targets = rewards + self.discount_factor * np.amax(successor_qvalues, axis=-1)

        # use the q target to update the optimising network
        self.optimising_network.train_step(inputs, targets, actions_one_hot)


# print('imported file')
