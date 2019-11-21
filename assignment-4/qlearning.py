"https://github.com/simoninithomas/Deep_reinforcement_learning_Course"
"https://github.com/simoninithomas/Deep_reinforcement_learning_Course.git"
import numpy as np
import gym
import random
env = gym.make("FrozenLake-v0")
action_size = env.action_space.n
state_size = env.observation_space.n

# initialize q-table
qtable = np.zeros((state_size, action_size))

# create hyper parameters
total_episodes = 5000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = .6                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005             # Exponential decay rate for exploration prob

rewards = []
counter=0
# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    counter +=1
    print(counter)
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