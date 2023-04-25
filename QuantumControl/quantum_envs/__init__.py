from gymnasium.envs.registration import register

register(
    id="quantum_envs/CNOTGateCalibration-v0",
    entry_point="quantum_envs.envs:CNOTGateCalibrationEnvironment",
    max_episode_steps=10000000,
)