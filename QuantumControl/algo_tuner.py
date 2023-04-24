import optuna

from cleanrl_utils.tuner import Tuner

import quantum_envs

### Try to always use a "direct_algo.py" as the script to avoid excessive registration times ###

tuner = Tuner(
    script="direct_ppo.py",
    metric="charts/episodic_return",
    metric_last_n_average_window=50,
    direction="maximize",
    aggregation_type="average",
    target_scores={
        "quantum_envs/QuantumGateCalibration-v0": [0, 5],
    },
    params_fn=lambda trial: {
        #"learning-rate": trial.suggest_loguniform("learning-rate", 0.0003, 0.003),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [1, 2, 4]),
        "update-epochs": trial.suggest_categorical("update-epochs", [1, 2, 4, 8]),
        "num-steps": trial.suggest_categorical("num-steps", [5, 16, 32, 64, 128]),
        #"vf-coef": trial.suggest_uniform("vf-coef", 0, 5),
        #"max-grad-norm": trial.suggest_uniform("max-grad-norm", 0, 5),
        #"total-timesteps": 100000,
        #"num-envs": 2,
    },
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
)
tuner.tune(
    num_trials=100,
    num_seeds=3,
)
