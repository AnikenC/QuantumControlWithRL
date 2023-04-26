# QuantumControlWithRL
Testing out the use of Reinforcement Learning (RL) for Quantum Control, in particular focusing on the task of Circuit-Level and Pulse-Level Gate Calibration.
---
## Getting Started
Most of the RL Algos are implemented through CleanRL, which provides many single-file implementations for testing. For this reason cleanrl_modules is included for reference to easily add alternate algorithms for testing. To start using RL Algorithms for Quantum Control tasks:
0. You should have poetry installed for managing package dependencies, and running inside a virtual environment with Python <3.10 for CleanRL compatability
1. First git clone the repository into a local folder
2. In the cleanrl_modules folder, run `poetry install` from the terminal to get all CleanRL Dependencies installed
3. In the root folder, run `poetry init` and then `poetry install` to install all other dependencies like `qiskit` and `qiskit_ibm_runtime`
4. To start training, run `python3 QuantumControl/ppo.py` on the command line from the root folder.

To use one of the registered quantum environments during training, just modify the environment id to the updated one. For example you can run `python3 QuantumControl/ppo.py --env-id="quantum_envs/QuantumGateCalibration-v0"` from the command line in the root folder.

The general structure for this repository is that RL algos can be taken from the `cleanrl_modules` folder, and then placed inside the `QuantumControl` folder. From there, they can be run as normal with a registered quantum environment directly, without any code changes if desired.

To access and make your own quantum environments, you need to make a Gymnasium compatible environment, and follow [Gymnasium Registration protocol for making a custom environment](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#make-your-own-custom-environment). The broad structure for creating a new environment is to put all relevant files for the environment in a folder in QuantumControl->quantum_envs->envs. Then update the  `__init__.py` file in QuantumControl->quantum_envs->envs by importing the new Gymnasium Environment Class created. Then register this new environment in the `__init__.py` file in QuantumControl->quantum_envs with the necessary registration details.

---
## To-Dos
1. Verify All Environments have Optimized Solutions, Simulate the Optimal Actions using the `taster.py` file
2. Make Environments OOP-ed by making Gate Calibration Classes, and then using them to create several relevant Gate Calibration Environments for Command-Line Level Customization
3. Adding State Preparation Environment, Single-Qubit X-Gate Calibration Environment, and Pulse-Level Two-Qubit Gate Calibration with Jittable Qiskit-Dynamics
4. Make the TD3, SAC, and TD3-Jax Algorithms Gymnasium-Friendly
5. Perform Hyperparameter Optimization with the `hyperparam_tuning.py` files for PPO, Batched PPO, SAC, TD3, RPO, TD3-Jax, and benchmark new training performance
6. Verify that Optuna supports Optimization while using GPUs, to speed up hyperameter tuning
