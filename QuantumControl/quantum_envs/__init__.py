from gymnasium.envs.registration import register

register(
    id="quantum_envs/CNOTGateCalibration-v0",
    entry_point="quantum_envs.envs:CNOTGateCalibrationEnvironment_V0",
    max_episode_steps=10000000,
)

register(
    id="quantum_envs/CNOTGateCalibration-v1",
    entry_point="quantum_envs.envs:CNOTGateCalibrationEnvironment_V1",
    max_episode_steps=10000000,
)

register(
    id="quantum_envs/BatchedCNOTGateCalibration-v0",
    entry_point="quantum_envs.envs:BatchedCNOTGateCalibrationEnvironment_V0",
    max_episode_steps=10000000,
)