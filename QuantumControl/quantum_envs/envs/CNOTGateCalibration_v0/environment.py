import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from quantum_envs.envs.CNOTGateCalibration_v0.quantum_environment import QuantumEnvironment
from quantum_envs.envs.CNOTGateCalibration_v0.target import CNOT
from quantum_envs.envs.CNOTGateCalibration_v0.static import AbstractionLevel

### Version 0 ###
# Uses Direct Fidelity Estimation as Reward

class CNOTGateCalibrationEnvironment_V0(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.qenvironment = QuantumEnvironment(target=CNOT(), abstraction_level=AbstractionLevel.CIRCUIT)
        self.init_obs = np.ones(1, dtype=np.float64)
        self.n_actions = 7
        self.reward = 0.
        self.max_reward = 0.
        self.step_for_max_reward = 0
        self.episode_length = 0
        self.simple_sample = 4
        self.complete_tomography_state_size = len(self.qenvironment.target.input_states)
        self.process_fidelity = 0.
        self.average_fidelity = 0.

        self.action_space = Box(
            low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float64
        )
        self.observation_space = Box(
            low=-1., high=1., shape=(1,), dtype=np.float64
        )
    
    def _get_obs(self):
        observation = self.init_obs
        return observation
    
    def _get_info(self):
        info = {
            "mean reward": np.mean(self.reward),
            "episode length": self.episode_length,
            "max reward": self.max_reward,
            "step for max": self.step_for_max_reward,
            "average fidelity": self.average_fidelity,
            "process fidelity": self.process_fidelity,
        }
        return info
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        ### Single Length Episode ###
        self.episode_length += 1

        index = np.random.randint(self.complete_tomography_state_size)
        if not self.simple_sample == 0:
            index = np.random.randint(self.simple_sample)
        self.reward, average_fidelity, process_fidelity = self.qenvironment.perform_action_gate_cal(action, index) # Can support batched actions
        self.process_fidelity = process_fidelity
        self.average_fidelity = average_fidelity
        if np.max(self.reward) > self.max_reward:
            self.max_reward = np.max(self.reward)
        observation = self._get_obs()
        info = self._get_info()
        terminated = True

        if terminated:
            self.episode_length = 0
        if len(self.reward) == 1:
            self.reward = self.reward[0]
        return observation, self.reward, terminated, False, info