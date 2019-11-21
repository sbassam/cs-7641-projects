# import numpy as np, numpy.random as nr, gym
# import matplotlib.pyplot as plt
#
# from gym.envs.toy_text import FrozenLakeEnv
# env = FrozenLakeEnv()
# np.set_printoptions(precision=3)
#
# # Seed RNGs so you get the same printouts as me
# env.seed(0)
# from gym.spaces.space import seeding
# seeding.np_random(10)
# # Generate the episode
# env.reset()
# for t in range(100):
#     env.render()
#     a = env.action_space.sample()
#     ob, rew, done, _ = env.step(a)
#     if done:
#         break
# assert done
# env.render();
#
# class MDP(object):
#     def __init__(self, P, nS, nA, desc=None):
#         self.P = P # state transition and reward probabilities, explained below
#         self.nS = nS # number of states
#         self.nA = nA # number of actions
#         self.desc = desc # 2D array specifying what each grid cell means (used for plotting)
# # HG use unwrapped to access
# mdp = MDP( {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.unwrapped.P.items()}, env.unwrapped.nS, env.unwrapped.nA, env.unwrapped.desc)
#
#
# def value_iteration(mdp, gamma, nIt, grade_print=print):
#     """
#     Inputs:
#         mdp: MDP
#         gamma: discount factor
#         nIt: number of iterations, corresponding to n above
#     Outputs:
#         (value_functions, policies)
#
#     len(value_functions) == nIt+1 and len(policies) == nIt
#     """
#     grade_print("Iteration | max|V-Vprev| | # chg actions | V[0]")
#     grade_print("----------+--------------+---------------+---------")
#     Vs = [np.zeros(mdp.nS)]  # list of value functions contains the initial value function V^{(0)}, which is zero
#     pis = []
#     for it in range(nIt):
#         oldpi = pis[-1] if len(pis) > 0 else None  # \pi^{(it)} = Greedy[V^{(it-1)}]. Just used for printout
#         Vprev = Vs[-1]  # V^{(it)}
#
#         # Your code should fill in meaningful values for the following two variables
#         # pi: greedy policy for Vprev (not V),
#         #     corresponding to the math above: \pi^{(it)} = Greedy[V^{(it)}]
#         #     ** it needs to be numpy array of ints **
#         # V: bellman backup on Vprev
#         #     corresponding to the math above: V^{(it+1)} = T[V^{(it)}]
#         #     ** numpy array of floats **
#
#         V = np.zeros(mdp.nS)
#         pi = np.zeros(mdp.nS)
#         # for each state in the set of states
#         for state in mdp.P:
#             maxv = 0
#             # loop through all the actions in the state
#             for action in mdp.P[state]:
#                 v = 0
#                 for probability, nextstate, reward in mdp.P[state][action]:
#                     v += probability * (reward + gamma * Vprev[nextstate])
#                 # if this the largest value for this state, update
#                 if v > maxv:
#                     maxv = v
#                     # greedy policy
#                     pi[state] = action
#             # note above, avoid updating value function in place
#             V[state] = maxv
#
#         max_diff = np.abs(V - Vprev).max()
#         nChgActions = "N/A" if oldpi is None else (pi != oldpi).sum()
#         grade_print("%4i      | %6.5f      | %4s          | %5.3f" % (it, max_diff, nChgActions, V[0]))
#         Vs.append(V)
#         pis.append(pi)
#     return Vs, pis
#
#
# GAMMA = 0.95  # we'll be using this same value in subsequent problems
#
# # The following is the output of a correct implementation; when
# #   this code block is run, your implementation's print output will be
# #   compared with expected output.
# #   (incorrect line in red background with correct line printed side by side to help you debug)
# expected_output = """Iteration | max|V-Vprev| | # chg actions | V[0]
# ----------+--------------+---------------+---------
#    0      | 0.80000      |  N/A          | 0.000
#    1      | 0.60800      |    2          | 0.000
#    2      | 0.51984      |    2          | 0.000
#    3      | 0.39508      |    2          | 0.000
#    4      | 0.30026      |    2          | 0.000
#    5      | 0.25355      |    1          | 0.254
#    6      | 0.10478      |    0          | 0.345
#    7      | 0.09657      |    0          | 0.442
#    8      | 0.03656      |    0          | 0.478
#    9      | 0.02772      |    0          | 0.506
#   10      | 0.01111      |    0          | 0.517
#   11      | 0.00735      |    0          | 0.524
#   12      | 0.00310      |    0          | 0.527
#   13      | 0.00190      |    0          | 0.529
#   14      | 0.00083      |    0          | 0.530
#   15      | 0.00049      |    0          | 0.531
#   16      | 0.00022      |    0          | 0.531
#   17      | 0.00013      |    0          | 0.531
#   18      | 0.00006      |    0          | 0.531
#   19      | 0.00003      |    0          | 0.531"""
# Vs_VI, pis_VI = value_iteration(mdp, gamma=GAMMA, nIt=20)
#
# for (V, pi) in zip(Vs_VI, pis_VI):
#     plt.figure(figsize=(3,3))
#     plt.imshow(V.reshape(4,4), cmap='gray', interpolation='none', clim=(0,1))
#     ax = plt.gca()
#     ax.set_xticks(np.arange(5)-.5)
#     ax.set_yticks(np.arange(5)-.5)
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     Y, X = np.mgrid[0:4, 0:4]
#     a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(-1, 0)}
#     Pi = pi.reshape(4,4)
#     for y in range(4):
#         for x in range(4):
#             a = Pi[y, x]
#             u, v = a2uv[a]
#             plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1)
#             plt.text(x, y, str(env.unwrapped.desc[y,x].item().decode()),
#                      color='g', size=12,  verticalalignment='center',
#                      horizontalalignment='center', fontweight='bold')
#     plt.grid(color='b', lw=2, ls='-')
# plt.figure()
# plt.plot(Vs_VI)
# plt.title("Values of different states")
# plt.show()
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
import numpy as np
import gym
import random
env = gym.make("FrozenLake-v0")
action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))
print(qtable)
total_episodes = 15000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005             # Exponential decay rate for exploration prob
# List of rewards
rewards = []

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        total_rewards += reward

        # Our new state is state
        state = new_state

        # If done (if we're dead) : finish episode
        if done == True:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

print("Score over time: " + str(sum(rewards) / total_episodes))
print(qtable)


