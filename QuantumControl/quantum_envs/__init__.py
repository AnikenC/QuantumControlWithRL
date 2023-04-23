from gymnasium.envs.registration import register

register(
    id="quantum_envs/QuantumGateCalibration-v0",
    entry_point="quantum_envs.envs:GateCalibrationEnvironment",
    max_episode_steps=10000000,
)