import quantum_envs
import gymnasium as gym

env = gym.make("quantum_envs/CNOTGateCalibration-v0")
obs, info = env.reset()