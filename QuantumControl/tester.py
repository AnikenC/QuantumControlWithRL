import gymnasium as gym
import quantum_envs

env = gym.make("quantum_envs/QuantumGateCalibration-v0")
obs, info = env.reset()
print(info)