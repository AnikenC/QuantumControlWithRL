# QuantumControlWithRL
Testing out the use of Reinforcement Learning (RL) for Quantum Control, in particular focusing on the task of Circuit-Level and Pulse-Level Gate Calibration.
---
## Getting Started
Most of the RL Algos are implemented through CleanRL, which provides many single-file implementations for testing. For this reason cleanrl_modules is included for reference to easily add alternate algorithms for testing. To start using RL Algorithms for Quantum Control tasks:
0. You should have poetry installed for managing package dependencies, and running inside a virtual environment with Python <3.10 for CleanRL compatability
1. First git clone the repository into a local folder
2. In the cleanrl_modules folder, run `poetry install` from the terminal to get all CleanRL Dependencies installed
3. In the root folder, run `poetry init` and then `poetry install` to install all other dependencies like `qiskit` and `qiskit_ibm_runtime`
4. Go into any of the Quantum Control task-specific folders like **Gate Calibration** and run any of the `algo_train.py` files from the comand line to test them out!

If not already, make sure you're logged into Weights and Biases, and if you want to directly log to wandb, add the `--track` argument when running algos for real-time logging. For example, from the root folder you can run `python3 GateCalibration/ppo_train.py --track` for training!
---
## What's new?
Most of the environment details are similar to as constructed before, however now there is a gymnasium environment with the `environment.py` file, and there is a CleanRL version of ppo with the `algo_train.py` files like `ppo_train.py` and `batched_ppo_train.py`.
### 1. Gymnasium Environment
The custom gymnasium environment is very similar to the normal gym environment, except for three things.
1. You now have to import gymnasium with `import gymnasium as gym`
2. The reset function requires a few modifications to directly handle seeding
3. The step function now returns two extra booleans instead of just `dones`, they are `terminated` and `truncated`. For simplicity we can treat `terminated` as `dones`, and `truncated` to always be False.
4. I added `_get_info` and `_get_obs` functions to make it a bit easier to read, and so that code doesn't have to be repeated with the reset and step functions for more complex environments
### 2. CleanRL Files
While CleanRL can handle registered Gymansium Environments, a few tweaks are needed from the `ppo_continuous.py` files they have in their module compared to the `ppo_train.py` we use, especially when the Gymnasium Environment is unregistered and only a class
1. Import your environment class from the relevant file
2. In the `make_env()` function, the standard environment creation with `env = gym.make("env-id")` needs to be replaced with `env = Environment()`
3. Based on the environment details, some of the `writer.add_scalar()` and print state statements have to be changed to handle the new info dictionary returned by the environment.
---
## Next Steps
Currently the `ppo_train.py` converges for small sample input sizes like two states, but doesn't converge for larger input sizes like the tomographically complete 16 states. The `batched_ppo_train.py` runs without errors, but convergence hasn't been reached yet for even small sample input states. Work needs to be done to achieve learning with the batched algorithm, achieve full input state learning as expected, and also perform hyperparameter optimization with the `tuner_example.py` files provided by CleanRL to improve learning.
