# from collections import defaultdict
#
# from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
# import numpy as np

import frozen_qlearning
from MDP import MDP
import gym


# set up the environment
#from plotting import plot_gw, plot_state_values_vs_iteration


# env = gym.make('FrozenLake8x8-v0')
#env = gym.make('FrozenLake-v0')
# from misc import  FrozenLakeEnv
#env = FrozenLakeEnv()

########
from gym.envs.toy_text.frozen_lake import generate_random_map
# https://reinforcementlearning4.fun/2019/06/24/create-frozen-lake-random-maps/

# random_map = generate_random_map(size=25, p=0.8)
#
# env = gym.make("FrozenLake-v0", desc=random_map)
#######
from plotting import plot_results

env = gym.make('FrozenLakeModified30x30-v0', is_slippery=False)
frozen_lake_VI_csv_list =[]
frozen_lake_PI_csv_list = []
frozen_lake_QL_csv_list = []
# instantiate the mdp
mdp = MDP({s: {a: [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS,
          env.nA, env.desc)

GAMMA = [0.80, 0.9, 0.95, 0.99]

for gamma in GAMMA:
    # perform greedy value iteration
    Vs_VI, pis_VI, csv_VI = mdp.value_iteration(gamma=gamma, epsilon=0.001, max_iter=200)
    frozen_lake_VI_csv_list.append(csv_VI)

    # perform policy iteration
    Vs_PI, pis_PI, csv_PI = mdp.policy_iteration(gamma=gamma, epsilon=0.001, max_iter=200)
    frozen_lake_PI_csv_list.append(csv_PI)


# perform Q-Learning
# gamma = 0.999
# epsilon = .5
# decay_rate = .999
# decay_type = 1
frozen_lake_QL_csv_list.append(
    frozen_qlearning.run_frozen_ql(epsilon=.5, decay_type=1, decay_rate=.999, gamma=.999, total_episodes=20000))
frozen_lake_QL_csv_list.append(
    frozen_qlearning.run_frozen_ql(epsilon=.85, decay_type=2, decay_rate=.999, gamma=.999, total_episodes=20000))
frozen_lake_QL_csv_list.append(
    frozen_qlearning.run_frozen_ql(epsilon=.5, decay_type=3, decay_rate=.999, gamma=.999, total_episodes=20000))
frozen_lake_QL_csv_list.append(
    frozen_qlearning.run_frozen_ql(epsilon=.85, decay_type=4, decay_rate=.999, gamma=.999, total_episodes=20000))

# plotting
plot_results('Frozen Lake 30x30', 'Value Iteration', frozen_lake_VI_csv_list, 'Iteration',
             'V_average', 'Iteration', 'Average reward', label_col = 'gamma', show_convergence=True)
plot_results('Frozen Lake 30x30', 'Value Iteration', frozen_lake_VI_csv_list, 'Iteration',
             'clock_time', 'Iteration', 'Time', label_col = 'gamma', show_convergence=True)
plot_results('Frozen Lake 30x30', 'Policy Iteration', frozen_lake_PI_csv_list, 'Iteration',
             'V_average', 'Iteration', 'Average reward', label_col = 'gamma', show_convergence=True)
plot_results('Frozen Lake 30x30', 'Policy Iteration', frozen_lake_PI_csv_list, 'Iteration',
             'clock_time', 'Iteration', 'Time', label_col = 'gamma', show_convergence=True)
plot_results('Frozen Lake 30x30', 'Q-Learning', frozen_lake_QL_csv_list, 'episode',
             'epsilon', 'Episode', 'Epsilon', label_col = 'epsilon', show_convergence=True, vertical_x=True)
plot_results('Frozen Lake 30x30', 'Q-Learning', frozen_lake_QL_csv_list, 'episode',
             'avg_reward', 'Episode', 'Average reward', label_col = 'decay', show_convergence=True, vertical_x=True)
plot_results('Frozen Lake 30x30', 'Q-Learning', frozen_lake_QL_csv_list, 'episode',
             'cum_discounted_rewards', 'Episode', 'Accumulated Discounted Reward', label_col = 'decay',
             show_convergence=True, vertical_x=True)
plot_results('Frozen Lake 30x30', 'Q-Learning', frozen_lake_QL_csv_list, 'episode',
             'accumulated_time', 'Episode', 'Time', label_col = 'decay', show_convergence=True, vertical_x=True)



