# imports
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# load environment and reset tf parameters
env = gym.make('FrozenLake-v0')
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

# implement neural network

# initialise input and vector layers
input_layer = tf.compat.v1.placeholder(shape=[1, 16], dtype=tf.float32)
output_layer = tf.Variable(tf.random.uniform([16, 4], 0, 0.01))
Q_values = tf.matmul(input_layer, output_layer)

prediction = tf.argmax(Q_values, 1)

# calculate loss and initialise gradient descent trainer
Q_target = tf.compat.v1.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Q_target - Q_values))
gradient_descent = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
perform_update = gradient_descent.minimize(loss)

# training loop

# initialise variables and parameter
init = tf.compat.v1.initialize_all_variables()
gamma = 0.99
epsilon = 0.1
num_episodes = 2000
step_array = []
reward_array = []

with tf.compat.v1.Session() as session:
    session.run(init)

    # loop through episodes
    for ep in range(num_episodes):
        # reset environment/variables and get new initial state
        state = env.reset()
        total_reward = 0
        time_step = 0

        # loop through time steps, training network
        while time_step < 99:
            time_step += 1

            # choose action epsilon-greedily from the Q-network value outputs from corresponding state
            action, action_Qs = session.run([prediction, Q_values], feed_dict={input_layer: np.identity(16)[state: state + 1]})
            if np.random.rand(1) < epsilon:
                action[0] = env.action_space.sample()

            # sample successor state, reward and terminal bool from environment
            successor, reward, terminal, _ = env.step(action[0])

            # retrieve new action Q-values from network on input of successor state
            successor_Q_values = session.run(Q_values, feed_dict={input_layer:np.identity(16)[successor: successor + 1]})

            # find the maximal successor action and retrieve its Q-value
            successor_Qmax = np.max(successor_Q_values)

            # construct the Q-target
            Q_target2 = successor_Q_values
            Q_target2[0, action[0]] = reward + gamma * successor_Qmax

            # train network using Q-target and predicted Q-values
            _, output_layer2 = session.run([perform_update, output_layer], feed_dict={input_layer: np.identity(16)[state: state+1], Q_target: Q_target2})

            # update variables before next pass
            total_reward += reward
            state = successor

            if terminal:
                # reduce epsilon value after each episode
                epsilon = 1/((ep / 50) + 10)
                break

        # save no. steps and total reward for analytics
        step_array.append(time_step)
        reward_array.append(reward)

print("Percent of successful episodes: " + str(sum(reward_array) / num_episodes))

