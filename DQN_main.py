"""
DQN Example
pip install gymnasium
pip install gymnasium[classic-control]

"""

import numpy as np
import torch
from torch.optim import Adam
from DQN_memory import MyMemory
from DQN import DQN
from peg_solitaire_env import Peg_Board
import random
from plot_board2 import plot_board


# training parameters
batch_size = 8192

# Environment
# toggle on/off render_mode for visualization
#env = gym.make('Acrobot-v1')#, render_mode="human")

env = Peg_Board()
action_space = env.action_space()

# Agent
#agent = DQN(env.observation_space.shape[0], env.action_space.n)
agent = DQN(49,76)
# agent = torch.load('model.pth')

# Memory
memory = MyMemory()
memory = memory
total_steps = 0
totalreward = 0
reward_eps = []
final_rewards = []
pegs_left = []
tot_steps = []
eps = []

path = "C:/Personal_Data/VT SEM2/Human Robot Interaction/New Folder_2/New Folder/images/"

# for i in range(17,23):
#     print(i)
#     for j in range(10000):
#         board = env.gen_rand_state(i)
#         moves = env.valid_moves(board)
#         if len(moves)==0:
#             continue
#         action = moves[random.randint(0,len(moves)-1)]
#         actionid = env.action_space().index(action)
#         next_state, reward, done = env.apply_move(board, action)
#         memory.push(board.reshape(25), actionid, reward, next_state.reshape(25), done)

# Main loop
for i_episode in range(1, 8000):
    # eps_steps=0
    episode_reward = 0
    done = False

    state = env.reset()
    while not done:
        # eps_steps += 1
        state2 = state.reshape(1, 49)
        # train the models
        if len(memory) > batch_size:
                loss = agent.update_parameters(memory, batch_size)
        # use policy to select action
        # choose random action for a while
        temp_valid_moves = env.valid_moves(state)
        while True:
            if i_episode < 3500:
                actionid = random.randint(0,75)
                action = action_space[actionid]
                if action in temp_valid_moves:
                    break
            # elif random.randint(0,10) < 4:
            elif i_episode < 5000 and np.random.rand() < 0.02:
            # elif i_episode %2 == 0:
                actionid = random.randint(0,75)
                action = action_space[actionid]
                if action in temp_valid_moves:
                    break
            # then use the learned policy
            else:
                Q_state = agent.select_action(state2)
                actionid = np.argmax(Q_state)
                action = action_space[actionid]
                while action not in temp_valid_moves:
                    Q_state[0][actionid] = -np.inf
                    actionid = np.argmax(Q_state)
                    action = action_space[actionid]
                break
        # transition to next_state and store data
        next_state, reward, done = env.apply_move(state,action)
        episode_reward += reward
        if [state.reshape(49), actionid, reward, next_state.reshape(49), done] not in memory.buffer:
            memory.push(state.reshape(49), actionid, reward, next_state.reshape(49), done)
        state = next_state
        total_steps += 1
    reward_eps.append(episode_reward)
    eps.append(i_episode)
    final_rewards.append(reward)
    tot_steps.append(total_steps)
    if i_episode == 1:
        pegs_left.append(33 - total_steps)
    else:
        pg_left = 33 - (tot_steps[i_episode - 1] - tot_steps[i_episode - 2])
        pegs_left.append(pg_left)

    totalreward += episode_reward
    print("memory buffer: ", len(memory.buffer))
    print("Episode: {}, Reward: {}".format(i_episode, round(episode_reward, 2)))
    print("Episode: {}, Final Reward: {}".format(i_episode, round(reward, 2)))


moving_averages = []
window_size = 200
i = 0
# Loop through the array to consider
# every window of size 100
while i < len(pegs_left) - window_size + 1:
    # Store elements from i to i+window_size
    # in list to get the current window
    window = pegs_left[i: i + window_size]

    # Calculate the average of current window
    window_average = round(sum(window) / window_size, 2)

    # Store the average of current
    # window in moving average list
    moving_averages.append(window_average)

    # Shift window to right by one position
    i += 1

moving_averages_reward = []
j = 0
# Loop through the array to consider
# every window of size 100
while j < len(final_rewards) - window_size + 1:
    # Store elements from i to i+window_size
    # in list to get the current window
    win = final_rewards[j: j + window_size]

    # Calculate the average of current window
    win_av = round(sum(win) / window_size, 2)

    # Store the average of current
    # window in moving average list
    moving_averages_reward.append(win_av)

    # Shift window to right by one position
    j += 1


import matplotlib.pyplot as plt
plt.plot(eps, pegs_left)
plt.title("Pegs Left")
plt.xlabel("Episode")
plt.ylabel("Pegs left")
plt.show()

plt.plot(eps, reward_eps)
plt.title("Episode reward")
plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.show()

plt.plot(eps[199:], moving_averages)
plt.title("Moving averages of pegs left at final state")
plt.xlabel("Episode")
plt.ylabel("Moving average")
plt.show()

plt.plot(eps[199:], moving_averages_reward)
plt.title("Moving averages of reward at final state")
plt.xlabel("Episode")
plt.ylabel("Moving average")
plt.show()
# print("average Reward: {}".format(round(totalreward/1000, 2)))
import pandas as pd

torch.save(agent, 'model_77_add_re.pth')

DF = pd.DataFrame(pegs_left)

# save the dataframe as a csv file
DF.to_csv("data_77_reward.csv")
