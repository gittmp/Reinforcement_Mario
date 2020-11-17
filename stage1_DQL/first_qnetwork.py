import numpy as np
import tensorflow as tf


# create variable for weights of NN layer
def param_tensor(shape, initialiser):
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

        # construct shape of weights for all connections between each layer
        weight_shapes = [
            # input layer -> 1st hidden layer
            [self.input_size, self.hidden_size[0]],
            # 1st hidden -> 2nd hidden
            [self.hidden_size[0], self.hidden_size[1]],
            # 2nd hidden -> output layer
            [self.hidden_size[1], self.output_size]
        ]

        # construct shape of biases for each neuron in hidden layers + output layer
        bias_shapes = [
            # 1st hidden layer neurons
            [1, self.hidden_size[0]],
            # 2nd hidden layer neurons
            [1, self.hidden_size[1]],
            # output layer neurons
            [1, self.output_size]
        ]

        # create tensors of the shape of weights/biases for each layer, and initialise each
        # biases init as 0s, weights along uniform distribution [-LIMIT, LIMIT]; LIMIT=sqrt(6/(no_inputs+no_outputs))
        self.weights = [param_tensor(conn, weights_initialiser) for conn in weight_shapes]
        self.biases = [param_tensor(neuron, bias_initialiser) for neuron in bias_shapes]
        self.trainable_vars = self.weights + self.biases

        # set network optimiser to inputted optimiser (default uses ADAM)
        self.optimiser = optimiser(**optimiser_kwargs)

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
        # watch tf variables (weights/biases) so that we can compute their gradient later
        with tf.GradientTape() as tape:
            # get action value outputs of network on that input (and squeeze to 1D array)
            qvalues = tf.squeeze(self.model(inputs))
            # use one hot action vector to mask q values to only the relevant action
            onehot_qvalues = tf.reduce_sum(qvalues * actions_one_hot, axis=1)
            # calculate loss function as the MSE between target output and these masked one hot q values
            mse_loss = tf.losses.mean_squared_error(targets, onehot_qvalues)

        # calculate the gradient along which we should update network weights to minimise the MSE loss
        gradients = tape.gradient(mse_loss, self.trainable_vars)

        # apply this gradient descent update to the network variables
        self.optimiser.apply_gradients(zip(gradients, self.trainable_vars))
