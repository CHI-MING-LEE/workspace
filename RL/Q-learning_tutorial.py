'''
@ Date: 2018/08/08
@ Ref: http://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
@ Topic: Reinforcement learning tutorial using Python and Keras

'''

import gym  # Open AI Gym Python package
import numpy as np

# A simple demo
# (state, reward, done)
env = gym.make("NChain-v0")
env.reset()
env.step(0)
env.step(1)

"""
state s will be initialized
1. machine perform an action (a1, a2, ...) to environment
2. interpreter views the action and feeds back an updated state (new_s) and the reward
"""


# naive approach
def naive_sum_reward_agent(env, num_episodes=20):
    # this is the table that will hold our summated rewards for
    # Each action in each state
    r_table = np.zeros((5, 2))
    for g in range(num_episodes):
        # g = 0
        # initialize s
        s = env.reset()
        done = False
        # So, the value r(s0,a0) would be, say,
        # the sum of the rewards that the agent has
        # received when in the past they have been in state 0 and taken action 0
        while not done:
            if np.sum(r_table[s, :]) == 0:
                # make a random selection of actions
                a = np.random.randint(0, 2)
            else:
                # select the action with highest cumulative reward
                a = np.argmax(r_table[s, :])
            new_s, r, done, _ = env.step(a)
            r_table[s, a] += r
            s = new_s
    return r_table


naive_sum_reward_agent(env=env, num_episodes=500)


def q_learning_with_table(env, num_episodes=500):
    q_table = np.zeros((5, 2))
    y = 0.95
    lr = 0.8
    for i in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            if np.sum(q_table[s, :]) == 0:
                # make a random selection of actions
                a = np.random.randint(0, 2)
            else:
                # select the action with largest q value in state s
                a = np.argmax(q_table[s, :])
            new_s, r, done, _ = env.step(a)
            # The first term, r, is the reward that was obtained when action a was taken in state s.
            q_table[s, a] += lr * (r + y * np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s
    return q_table


q_learning_with_table(env)

"""
The ϵ-greedy policy in reinforcement learning is basically the same as the greedy policy, except that there is a value ϵ 
(which may be set to decay over time) where, if a random number is selected which is less than this value, an action is 
chosen completely at random. This step allows some random exploration of the value of various actions in various states,
and can be scaled back over time to allow the algorithm to concentrate more on exploiting the best strategies that it
has found. This mechanism can be expressed in code as:
"""


def eps_greedy_q_learning_with_table(env, num_episodes=500):
    q_table = np.zeros((5, 2))
    y = 0.95
    eps = 0.5
    lr = 0.8
    decay_factor = 0.999
    for i in range(num_episodes):
        """Initialization"""
        s = env.reset()
        eps *= decay_factor
        done = False
        while not done:
            """choose a from s using policy derived from Q(e.g., ϵ-greedy)"""
            if np.random.random() < eps or np.sum(q_table[s, :]) == 0:
                # make a random selection of actions
                a = np.random.randint(0, 2)
            else:
                # select the action with largest q value in state s
                a = np.argmax(q_table[s, :])
            """Take action a, observ r, s'"""
            new_s, r, done, _ = env.step(a)
            # 如果這邊再用一個ϵ-greedy選new_a(a_2)，那就是Sarsa(On-Policy)的策略；反之，這裡是Q-learning
            # The first term, r, is the reward that was obtained when action a was taken in state s.
            q_table[s, a] += lr * (r + y * np.max(q_table[new_s, :]) - q_table[s, a])
            """Update s"""
            s = new_s
    return q_table


eps_greedy_q_learning_with_table(env)

"""Comparing the methods"""

"""
Here, it can be observed that the trained table given to the function is used for action selection, 
and the total reward accumulated during the game is returned.
"""


def run_game(table, env):
    s = env.reset()
    tot_reward = 0
    done = False
    while not done:
        a = np.argmax(table[s, :])
        s, r, done, _ = env.step(a)
        tot_reward += r
    return tot_reward


def test_methods(env, num_iterations=100):
    winner = np.zeros((3,))
    for g in range(num_iterations):
        # first make a Q-table
        m0_table = naive_sum_reward_agent(env, 500)
        m1_table = q_learning_with_table(env, 500)
        m2_table = eps_greedy_q_learning_with_table(env, 500)
        # get the total reward of an episode
        m0 = run_game(m0_table, env)
        m1 = run_game(m1_table, env)
        m2 = run_game(m2_table, env)
        w = np.argmax(np.array([m0, m1, m2]))
        winner[w] += 1
        print(f"Game {g + 1} of {num_iterations}")
    return winner


test_methods(env)

"""
Instead of having explicit tables, instead we can train a neural network to 
predict Q values for each action in a given state.
"""

from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense

model = Sequential()
model.add(InputLayer(batch_input_shape=(1, 5)))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()

"""
Inputs corresponding to the one-hot encoded state vectors:
[0, 0, 0, 1, 0] -> means in state 3

loss = (r + y*max Q'(s', a') - Q(s, a))**2

(target - prediction)**2: L2 loss

"""

# now execute the q learning
y = 0.95
eps = 0.5
decay_factor = 0.999
num_episodes = 100
r_avg_list = []
for i in range(num_episodes):
    s = env.reset()
    eps *= decay_factor
    if i % 10 == 0:
        print(f"Episode {i + 1} of {num_episodes}")
    done = False
    r_sum = 0
    while not done:
        # ϵ -greedy action
        if np.random.random() < eps:
            a = np.random.randint(0, 2)
        else:
            # predict Q-values & choose action a
            a = np.argmax(model.predict(np.identity(5)[s:s + 1]))
        new_s, r, done, _ = env.step(a)
        # Update the target value (當前reward + 潛在最大可能reward)
        target = r + y * np.max(model.predict(np.identity(5)[new_s:new_s + 1]))
        # the output of neural network (two Q-values)
        target_vec = model.predict(np.identity(5)[s:s + 1])[0]
        # only change Q-value of the chosen action (Q-value是選擇動作a的分數)
        target_vec[a] = target
        # 讓target盡量與NN的output相同 -> 收斂Q-values
        model.fit(np.identity(5)[s:s + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)
        s = new_s
        r_sum += r
    r_avg_list.append(r_sum / 1000)  # 1000 games for each episode

model.predict(np.identity(5))
