# QuantumControlWithRL
Testing out the use of Reinforcement Learning (RL) for Quantum Control, in particular focusing on the task of Circuit-Level and Pulse-Level Gate Calibration.

## Getting Started
Most of the RL Algos are implemented through CleanRL, which provides many single-file implementations for testing. For this reason cleanrl_modules is included for reference to easily add alternate algorithms for testing. To start using RL Algorithms for Quantum Control tasks:
0. You should have poetry installed for managing package dependencies, and running inside a virtual environment with Python <3.10 for CleanRL compatability
1. First git clone the repository into a local folder
2. In the cleanrl_modules folder, run `poetry install` from the terminal to get all CleanRL Dependencies installed
3. In the root folder, run `poetry init` and then `poetry install` to install all other dependencies like `qiskit` and `qiskit_ibm_runtime`
4. Go into any of the Quantum Control task-specific folders like **Gate Calibration** and run any of the `algo_train.py` files from the comand line to test them out!

If not already, make sure you're logged into Weights and Biases, and if you want to directly log to wandb, add the `--track` argument when running algos for real-time logging. For example, from the root folder you can run `python3 GateCalibration/ppo_train.py --track` for training!
