from collections import defaultdict

from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np
from MDP import MDP
import gym
import misc

# set up the environment
from plotting import plot_gw, plot_state_values_vs_iteration

env = gym.make('FrozenLake8x8-v0')
#env = gym.make('FrozenLake-v0')
# from misc import  FrozenLakeEnv
#env = FrozenLakeEnv()

########
from gym.envs.toy_text.frozen_lake import generate_random_map
# https://reinforcementlearning4.fun/2019/06/24/create-frozen-lake-random-maps/

random_map = generate_random_map(size=25, p=0.8)

env = gym.make("FrozenLake-v0", desc=random_map)
#######

# instantiate the mdp
mdp = MDP({s: {a: [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS,
          env.nA, env.desc)
GAMMA = 0.95
# perform greedy value iteration
Vs_VI, pis_VI = mdp.value_iteration(gamma=GAMMA, epsilon=0.001, max_iter=200)
# print (Vs_VI, pis_VI)
# #
# plot_gw(env, GAMMA, Vs_VI, pis_VI, 'value-iteration')
# plot_state_values_vs_iteration(GAMMA, Vs_VI, 'value-iteration')
#
# # perform policy iteration
Vs_PI, pis_PI = mdp.policy_iteration(gamma=GAMMA, epsilon=0.001, max_iter=200)
# plot_gw(env, GAMMA, Vs_PI, pis_PI, 'policy-iteration')
plot_state_values_vs_iteration(GAMMA, Vs_PI, 'policy-iteration')

# perform Q-Learning


