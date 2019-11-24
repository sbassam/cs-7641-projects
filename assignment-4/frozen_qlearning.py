"https://github.com/simoninithomas/Deep_reinforcement_learning_Course"
import pandas as pd
from gym.envs.toy_text.frozen_lake import generate_random_map

"https://github.com/simoninithomas/Deep_reinforcement_learning_Course.git"
import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import time

env = gym.make("FrozenLake-v0")

env = gym.make('FrozenLakeModified8x8-v0')
# env = gym.make('FrozenLakeModified30x30-v0', is_slippery=False)
# random_map = generate_random_map(size=30, p=0.8)
# env = gym.make("FrozenLakeModified30x30-v0", desc=random_map, is_slippery=False)

# random_map = generate_random_map(size=30, p=0.8)
# env = gym.make("FrozenLakeModified30x30-v0", desc=random_map, is_slippery=True)
# env = gym.make("FrozenLake-v0", desc=random_map, is_slippery=True)
from gym.envs.registration import register


# register(
#     id='FrozenLake15x15-v1',
#     entry_point='frozen_lake_custom:FrozenLakeCustom',
#     max_episode_steps=10000,
#     reward_threshold=0.99,  # optimum = 1
#     kwargs={'desc': random_map, 'goalr': 5.0, 'fallr': -1.0, 'stepr': -0.04}
# )
# env = gym.make("FrozenLake15x15-v1")


def choose_action(epsilon, Q, state):
    """inspired from https://gym.openai.com/evaluations/eval_xSOlwrBsQDqUW7y6lJOevQ/"""
    explore = False
    action = np.argmax(Q[state, :])

    if np.random.rand() < epsilon:
        action = np.random.randint(Q.shape[1])
        explore = True
    return action, explore


csv_path = 'out/ql-' + env.spec._env_name + '.csv'
data = []
cols = ['episode', 'epsilon', 'terminal_reward', 'cum_discounted_rewards', 'episode_tot_rewards', 'avg_reward',
        'avg_delta_reward', 'gamma', 'alpha', 'episode_time','accumulated_time', 'num_exploitation', 'num_exploration',
        'done?', 'steps_before_done', 'state_size', 'decay_rate', 'decay_type']

action_size = env.action_space.n
state_size = env.observation_space.n

# best params for 8x8
# max_steps = 500
# gamma = 0.95
# epsilon = .02
#
#
# reward_tracker = [] #taken from https://learning.oreilly.com/videos/reinforcement-learning-and/9781491995006/9781491995006-video312886
# qtable = np.zeros((state_size, action_size))
# total_episodes = 5000
# alpha = .8
# decaytype == 1
####################

max_steps = 500
gamma = 0.98
epsilon = .5
decay_rate = .999
decay_type = 1

reward_tracker = []  # taken from https://learning.oreilly.com/videos/reinforcement-learning-and/9781491995006/9781491995006-video312886
reward_diffs = []
qtable = np.zeros((state_size, action_size))
total_episodes = 5000
alpha = .8

counter = 0


# 2 For life or until learning is stopped
def decay_eps(epsilon, episode=None, decay_rate=0.9, decaytype=1):
    if decaytype == 1:
        epsilon = epsilon * decay_rate
    elif decaytype == 2:
        epsilon = 0.01 + (.99) * np.exp(-.005 * episode)

    return epsilon


cum_time = 0
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    counter += 1
    num_explore = 0
    num_exploit = 0
    cumu_r = 0
    episode_rewards = []
    episode_deltas = []

    start = time.time()
    for step in range(max_steps):
        action, explore = choose_action(epsilon, qtable, state)

        if explore:
            num_explore += 1
        else:
            num_exploit += 1
        new_state, reward, done, info = env.step(action)
        delta = reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action]
        qtable[state, action] = qtable[state, action] + alpha * delta
        episode_deltas.append(delta)
        episode_rewards.append(reward)
        total_rewards += reward
        cumu_r = reward + gamma * cumu_r
        state = new_state
        if done:
            break
    end = time.time()
    episode_time = end - start
    cum_time += episode_time
    # store the episode statistics
    row = [episode, epsilon, reward, cumu_r, total_rewards, np.mean(episode_rewards), np.mean(episode_deltas), gamma,
           alpha, episode_time,
           cum_time, num_exploit, num_explore, done, step, state_size, decay_rate, decay_type]
    data.append(row)
    # update epsilon
    epsilon = decay_eps(epsilon, episode, decay_rate=decay_rate, decaytype=decay_type)

result = pd.DataFrame(data, columns=cols)
result.to_csv(csv_path, index=None)

plt.plot(range(total_episodes), result['terminal_reward'])
plt.show()
plt.close()
plt.plot(range(total_episodes), result['cum_discounted_rewards'])
plt.show()   
plt.close()
plt.plot(range(total_episodes), result['episode_tot_rewards'])

plt.plot(range(total_episodes), result['avg_reward'])
plt.plot(range(total_episodes), result['avg_delta_reward'])
plt.show()


print("Score over time: " + str(sum(reward_tracker) / total_episodes))
print(qtable)

# from collections import defaultdict
# import random
#
# import numpy as np
# from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
#
# def eps_greedy(q_vals, eps, state):
#     # exploration vs. exploitation
#     if random.random() < eps:
#         action = np.random.choice(len(q_vals[state]))
#     else:
#         action = np.argmax(q_vals[state])
#     return action
#
# def q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward):
#     target = reward + gamma * np.max(q_vals[next_state])
#     q_vals[cur_state][action] = (1 - alpha) * q_vals[cur_state][action] + alpha * target
#


env.reset()

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):

        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state, :])

        new_state, reward, done, info = env.step(action)

        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()

            # We print the number of step it took.
            print("Number of steps", step)
            break
        state = new_state
env.close()
