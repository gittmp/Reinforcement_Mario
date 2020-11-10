# Deep Q-Learning with Experience Replay code example from 'Playing Atari with Deep Reinforcement Learning'

import gym
import numpy as np
import tensorflow as tf

# a) Initialise replay memory D with capacity N
# b) Initialise DNN Q with random weights as params theta
# c) For episodes 1 -> M
#     i. Initialise state s_t = s_0 and processed state phi_s_t = phi(s_0)
#     ii. For timesteps t 1 -> T
#         ACTING
#         1) With probability epsilon select random action
#         2) Else, run phi_s_t through Q and select maximal action - i.e. a_t = max( Q(phi_s_t; theta) )
#         3) Run a_t through the environment at state s_t to produce reward r_t, successor succ_t, terminal bool
#         4) Process sucessor state phi_succ_t = phi(succ_t)
#         5) Store tuple (phi_s_t, a_t, r_t, phi_succ_t) in D
#         TRAINING
#         6) Sample some random tuple (phi_s_j, a_j, r_j, phi_succ_j) from D
#         7) If phi_succ_j[terminal] == True, set the q_target to reward r_j
#         8) Else, q_target = r_j + gamma * max( Q(phi_succ_j; theta) )
#         9) Calculate loss function L = (q_target - Q(phi_s_j; theta)[a_j])^2
#         10) Find new parameters theta by performing gradient descent update minimising L through adjusting theta
