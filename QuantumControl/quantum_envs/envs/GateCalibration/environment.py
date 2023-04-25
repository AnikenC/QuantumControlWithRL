import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from quantum_envs.envs.GateCalibration.quantum_environment import QuantumEnvironment
from quantum_envs.envs.GateCalibration.target import CNOT
from quantum_envs.envs.GateCalibration.static import AbstractionLevel

class CNOTGateCalibrationEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.qenvironment = QuantumEnvironment(target=CNOT(), abstraction_level=AbstractionLevel.CIRCUIT)
        self.init_obs = np.ones(1, dtype=np.float64)
        self.n_actions = 7
        self.reward = 0.
        self.max_reward = 0.
        self.step_for_max_reward = 0
        self.episode_length = 0
        self.complete_tomography_state_size = len(self.qenvironment.target.input_states)

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
            "step for max": self.step_for_max_reward
        }
        return info
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action, simple_sample=0):
        ### Single Length Episode ###
        self.episode_length += 1

        assert simple_sample.type == int, "Only an integer number of input samples can be used"
        assert simple_sample > 0 and simple_sample < self.complete_tomography_state_size, f"Size must be greater than 0 and less than or equal to {self.complete_tomography_state_size}"
        index = np.random.randint(self.complete_tomography_state_size)
        if not self.simple_sample == 0:
            index = np.random.randint(self.simple_size)
        self.reward = self.qenvironment.perform_action_gate_cal(action, index) # Can support batched actions
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