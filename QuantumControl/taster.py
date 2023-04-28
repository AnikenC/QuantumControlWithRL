import time

import quantum_envs
import gymnasium as gym
import numpy as np

import matplotlib.pyplot as plt

### For Debugging the Environments and Verifying Results ###

env = gym.make("quantum_envs/CNOTGateCalibration-v1")
obs, info = env.reset()
ideal_action = 2 * np.array([ 0.00285997 , 0.34421486, -0.61353683, -0.25360537, -0.26386875,  0.25230902, 0.2414299], dtype=np.float64)
random_action = 0.2 * np.ones(7, dtype=np.float64)

sample_size = 10

reward_arr = np.zeros(sample_size)
prc_arr = np.zeros(sample_size)
avg_arr = np.zeros(sample_size)

max_reward = 0.
max_avg = 0.
max_prc = 0.

min_reward = 4.
min_avg = 4.
min_prc = 4.

start_time = time.time()


for i in range(sample_size):
    obs, reward, terminated, truncated, info = env.step(ideal_action)
    prc = info["process fidelity"]
    avg = info["average fidelity"]
    reward_arr[i] = reward
    avg_arr[i] = avg
    prc_arr[i] = prc
    if prc > max_prc:
        max_prc = prc
    elif prc < min_prc:
        min_prc = prc
    if avg > max_avg:
        max_avg = avg
    elif avg < min_avg:
        min_avg = avg
    if reward > max_reward:
        max_reward = reward
    elif reward < min_reward:
        min_reward = reward
    #print(f"Run {i}")
    #print(f"reward: {reward}")
    #print(f"process fidelity: {prc}")
    #print(f"average fidelity: {avg}")

end_time = time.time()
print(f"time taken: {end_time - start_time}, time taken per evaluation: {(end_time - start_time)/sample_size}")

mean_reward = np.mean(reward_arr)
mean_avg = np.mean(avg_arr)
mean_prc = np.mean(prc_arr)

std_reward = np.std(reward_arr, dtype=np.float64)
std_avg = np.std(avg_arr, dtype=np.float64)
std_prc = np.std(prc_arr, dtype=np.float64)

print("Evaluation Statistics")
print(f"mean reward: {mean_reward}, standard deviation: {std_reward}, max: {max_reward}, min: {min_reward}")
print(f"mean process fidelity: {mean_prc}, standard deviation: {std_prc}, max: {max_prc}, min: {min_prc}")
print(f"mean average fidelity: {mean_avg}, standard deviation: {std_avg} max: {max_avg}, min: {min_avg}")


'''
When we send an action (gate) into the environment, we want to get a good measure of its fidelity
To do this we can use Direct Fidelity Estimation along with a Tomographically Complete Set of 16 States
Currently, each time we send an action to the environment, we get an output that the one action 
would have done for one of the 16 states. But we need to average over 100 pauli shots, and all states

Solution:
Make the environment itself average over 300 repeats, then variance should majorly decrease
and the learning should become stable, we finally get an accurate fidelity estimate!!!!

'''