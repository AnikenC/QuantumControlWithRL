import time

import optuna
print(f"optuna version: {optuna.__version__}")

from cleanrl_utils.tuner import Tuner

import quantum_envs

### Make sure you specify the target scores environments carefully ###
### Check out https://docs.cleanrl.dev/advanced/hyperparameter-tuning/ for more details ###

script = "ppo.py"
algo_name = script.rstrip(".py")
env_id = "quantum_envs/CNOTGateCalibration-v0"
exp_name = f"tuner__{algo_name}__{env_id}__{int(time.time())}"

tuner = Tuner(
    script = script,
    metric="charts/episodic_return",
    metric_last_n_average_window=50,
    direction="maximize",
    aggregation_type="average",
    target_scores={
        env_id: [0, 5],
    },
    params_fn=lambda trial: {
        "learning-rate": trial.suggest_float("learning-rate", 0.0003, 0.03),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [1, 2, 4, 8]),
        "update-epochs": trial.suggest_categorical("update-epochs", [1, 2, 4, 8]),
        "num-steps": trial.suggest_categorical("num-steps", [5, 16, 32, 64, 128, 256]),
        "vf-coef": trial.suggest_float("vf-coef", 0, 5),
        "max-grad-norm": trial.suggest_float("max-grad-norm", 0, 5),
        "total-timesteps": 100000, # Determines how many total steps we train each environment in a trial for
        "num-envs": 1,
    },
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
    wandb_kwargs={"project": "GateCalibration",
                "entity": "quantumcontrolwithrl",
                "sync_tensorboard": True},
)
tuner.tune(
    num_trials=100,
    num_seeds=3,
)
