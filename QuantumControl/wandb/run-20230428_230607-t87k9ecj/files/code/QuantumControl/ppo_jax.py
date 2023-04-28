import time
import string
from typing import Callable
from tqdm.notebook import tqdm
from termcolor import colored
import signal

import numpy as np
from numpy import ndarray

import gymnasium as gym

import tensorflow as tf
import tensorflow_probability.substrates.jax.distributions as tfp

import jax.numpy as jnp
import jax.random as random
from jax.lax import stop_gradient

from jax import value_and_grad, jit, device_get, Array, random

import flax.linen as nn
from flax import struct
from flax.core import FrozenDict
from flax.struct import dataclass
from flax.training.train_state import TrainState
from flax.metrics.tensorboard import SummaryWriter
from flax.training import checkpoints

import optax
import wandb

import quantum_envs

tf.config.experimental.set_visible_devices([], 'GPU')

### Hyperparameters ###
# Need to be Argparsed
total_timesteps = 8000000 # total timesteps of the experiment
learning_rate = 3e-4 # the learning rate of the optimizer
num_envs = 1 # the number of parallel environments
num_steps = 300 # the number of steps to run in each environment per policy rollout
gamma = 0.99 # the discount factor gamma
gae_lambda = 0.95 # the lambda for the general advantage estimation
num_minibatches = 3 # the number of mini batches
update_epochs = 3 # the K epochs to update the policy
clip_coef = 0.2 # the surrogate clipping coefficient
ent_coef = 0.0 # coefficient of the entropy
vf_coef = 0.5 # coefficient of the value function
max_grad_norm = 0.5 # the maximum norm for the gradient clipping
seed = 1 # seed for reproducible benchmarks
exp_name = 'PPO' # unique experiment name
env_id= "quantum_envs/BatchedCNOTGateCalibration-v0" # id of the environment

batch_size = num_envs * num_steps # size of the batch after one rollout
minibatch_size = batch_size // num_minibatches # size of the mini batch
num_updates = total_timesteps // batch_size # the number of learning cycle

### Make Environment ###
def make_env(env_id: string, gamma: float):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

envs = gym.vector.SyncVectorEnv(
    [make_env(env_id, gamma) for i in range(num_envs)]
) # AsyncVectorEnv is faster, but we cannot extract single environment from it
assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
obs, _ = envs.reset()

### Create Agent Model ###
# Helper function to quickly declare linear layer with weight and bias initializers
def linear_layer_init(features, std=np.sqrt(2), bias_const=0.0):
    layer = nn.Dense(features=features, kernel_init=nn.initializers.orthogonal(std), bias_init=nn.initializers.constant(bias_const))
    return layer

from jax import Array
import jax.numpy as jnp

class Actor(nn.Module):
    action_shape_prod: int

    @nn.compact
    def __call__(self, x: Array):
        action_mean = nn.Sequential([
            linear_layer_init(64),
            nn.tanh,
            linear_layer_init(64),
            nn.tanh,
            linear_layer_init(self.action_shape_prod, std=0.01),
        ])(x)
        actor_logstd = self.param('logstd', nn.initializers.zeros, (1, self.action_shape_prod))
        action_logstd = jnp.broadcast_to(actor_logstd, action_mean.shape) # Make logstd the same shape as actions
        return action_mean, action_logstd

class Critic(nn.Module):
    @nn.compact
    def __call__(self, x: Array):
        return nn.Sequential([
            linear_layer_init(64),
            nn.tanh,
            linear_layer_init(64),
            nn.tanh,
            linear_layer_init(1, std=1.0),
        ])(x)

actor = Actor(action_shape_prod=np.array(envs.single_action_space.shape).prod()) # For jit we need to declare prod outside of class
critic = Critic()

### Create Agent State ###
# Setting seed of the environment for reproduction
key = random.PRNGKey(seed)
np.random.seed(seed)

key, actor_key, critic_key, action_key, permutation_key = random.split(key, num=5)

# Initializing agent parameters
actor_params = actor.init(actor_key, obs)
critic_params = critic.init(critic_key, obs)

# Anneal learning rate over time
def linear_schedule(count):
    frac = 1.0 - (count // (num_minibatches * update_epochs)) / num_updates
    return learning_rate * frac

tx = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.inject_hyperparams(optax.adamw)(
        learning_rate=linear_schedule,
        eps=1e-5
    )
)

@dataclass
class AgentParams:
    actor_params: FrozenDict
    critic_params: FrozenDict

# Probably jitting isn't needed as this functions should be jitted already
actor.apply = jit(actor.apply)
critic.apply = jit(critic.apply)

class AgentState(TrainState):
    # Setting default values for agent functions to make TrainState work in jitted function
    actor_fn: Callable = struct.field(pytree_node=False)
    critic_fn: Callable = struct.field(pytree_node=False)

agent_state = AgentState.create(
    params=AgentParams(
        actor_params=actor_params,
        critic_params=critic_params
    ),
    tx=tx,
    # As we have separated actor and critic we don't use apply_fn
    apply_fn=None,
    actor_fn=actor.apply,
    critic_fn=critic.apply
)

### Only Run this to continue training ###
tx = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.inject_hyperparams(optax.adamw)(
        learning_rate=linear_schedule,
        eps=1e-5
    )
)

agent_state = AgentState.create(
    params=AgentParams(
        actor_params=agent_state.params.actor_params,
        critic_params=agent_state.params.critic_params
    ),
    tx=tx,
    apply_fn=None,
    actor_fn=actor.apply,
    critic_fn=critic.apply
)


### Create Storage ###
@dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array

### Sample Action ###
@jit
def get_action_and_value(agent_state: AgentState, next_obs: ndarray, next_done: ndarray, storage: Storage, step: int, key: random.PRNGKeyArray):
    action_mean, action_logstd = agent_state.actor_fn(agent_state.params.actor_params, next_obs)
    value = agent_state.critic_fn(agent_state.params.critic_params, next_obs)
    action_std = jnp.exp(action_logstd)

    # Sample continuous actions from Normal distribution
    probs = tfp.Normal(action_mean, action_std)
    key, subkey = random.split(key)
    action = probs.sample(seed=subkey)
    logprob = probs.log_prob(action).sum(1)
    storage = storage.replace(
        obs=storage.obs.at[step].set(next_obs),
        dones=storage.dones.at[step].set(next_done),
        actions=storage.actions.at[step].set(action),
        logprobs=storage.logprobs.at[step].set(logprob),
        values=storage.values.at[step].set(value.squeeze()),
    )
    return storage, action, key

@jit
def get_action_and_value2(agent_state: AgentState, params: AgentParams, obs: ndarray, action: ndarray):
    action_mean, action_logstd = agent_state.actor_fn(params.actor_params, obs)
    value = agent_state.critic_fn(params.critic_params, obs)
    action_std = jnp.exp(action_logstd)

    probs = tfp.Normal(action_mean, action_std)
    return probs.log_prob(action).sum(1), probs.entropy().sum(1), value.squeeze()


### Rollout ###
def rollout(
        agent_state: AgentState,
        next_obs: ndarray,
        next_done: ndarray,
        storage: Storage,
        key: random.PRNGKeyArray,
        global_step: int,
        writer: SummaryWriter,
):
    for step in range(0, num_steps):
        global_step += 1 * num_envs
        storage, action, key = get_action_and_value(agent_state, next_obs, next_done, storage, step, key)
        next_obs, reward, terminated, truncated, infos = envs.step(device_get(action))
        next_done = terminated | truncated
        storage = storage.replace(rewards=storage.rewards.at[step].set(reward))

        # Only print when at least 1 env is done
        if "final_info" not in infos:
            continue

        for info in infos["final_info"]:
            # Skip the envs that are not done
            if info is None:
                continue
            writer.scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.scalar("charts/episodic_length", info["episode"]["l"], global_step)
    return next_obs, next_done, storage, key, global_step

### Compute GAE ###
@jit
def compute_gae(
        agent_state: AgentState,
        next_obs: ndarray,
        next_done: ndarray,
        storage: Storage
):
    # Reset advantages values
    storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
    next_value = agent_state.critic_fn(agent_state.params.critic_params, next_obs).squeeze()
    # Compute advantage using generalized advantage estimate
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - storage.dones[t + 1]
            nextvalues = storage.values[t + 1]
        delta = storage.rewards[t] + gamma * nextvalues * nextnonterminal - storage.values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        storage = storage.replace(advantages=storage.advantages.at[t].set(lastgaelam))
    # Save returns as advantages + values
    storage = storage.replace(returns=storage.advantages + storage.values)
    return storage

### PPO Loss ###
@jit
def ppo_loss(
        agent_state: AgentState,
        params: AgentParams,
        obs: ndarray,
        act: ndarray,
        logp: ndarray,
        adv: ndarray,
        ret: ndarray,
        val: ndarray,
):
    newlogprob, entropy, newvalue = get_action_and_value2(agent_state, params, obs, act)
    logratio = newlogprob - logp
    ratio = jnp.exp(logratio)

    # Calculate how much policy is changing
    approx_kl = ((ratio - 1) - logratio).mean()

    # Advantage normalization
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # Policy loss
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss
    v_loss_unclipped = (newvalue - ret) ** 2
    v_clipped = val + jnp.clip(
        newvalue - val,
        -clip_coef,
        clip_coef,
    )
    v_loss_clipped = (v_clipped - ret) ** 2
    v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
    v_loss = 0.5 * v_loss_max.mean()

    # Entropy loss
    entropy_loss = entropy.mean()

    # main loss as sum of each part loss
    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
    return loss, (pg_loss, v_loss, entropy_loss, stop_gradient(approx_kl))

### Update PPO ###
def update_ppo(
        agent_state: AgentState,
        storage: Storage,
        key: random.PRNGKeyArray
):
    # Flatten collected experiences
    b_obs = storage.obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = storage.logprobs.reshape(-1)
    b_actions = storage.actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = storage.advantages.reshape(-1)
    b_returns = storage.returns.reshape(-1)
    b_values = storage.values.reshape(-1)

    # Create function that will return gradient of the specified function
    ppo_loss_grad_fn = jit(value_and_grad(ppo_loss, argnums=1, has_aux=True))

    for epoch in range(update_epochs):
        key, subkey = random.split(key)
        b_inds = random.permutation(subkey, batch_size, independent=True)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                agent_state,
                agent_state.params,
                b_obs[mb_inds],
                b_actions[mb_inds],
                b_logprobs[mb_inds],
                b_advantages[mb_inds],
                b_returns[mb_inds],
                b_values[mb_inds],
            )
            # Update an agent
            agent_state = agent_state.apply_gradients(grads=grads)

    # Calculate how good an approximation of the return is the value function
    y_pred, y_true = b_values, b_returns
    var_y = jnp.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, explained_var, key

### Train Agent ###
# Make kernel interrupt be handled as normal python error
signal.signal(signal.SIGINT, signal.default_int_handler)

run_name = f"{exp_name}_{seed}_{time.asctime(time.localtime(time.time())).replace('  ', ' ').replace(' ', '_')}"
wandb.init(
    project="GateCalibration",
    entity="quantumcontrolwithrl",
    sync_tensorboard=True,
    name=run_name,
    save_code=True,
    config={
        'total_timesteps': total_timesteps,
        'learning_rate': learning_rate,
        'num_envs': num_envs,
        'num_steps': num_steps,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'num_minibatches': num_minibatches,
        'update_epochs': update_epochs,
        'clip_coef': clip_coef,
        'ent_coef': ent_coef,
        'vf_coef': vf_coef,
        'max_grad_norm': max_grad_norm,
        'seed': seed,
        'batch_size': batch_size,
        'minibatch_size': minibatch_size,
        'num_updates': num_updates,
    }
)
writer = SummaryWriter(f'runs/{env_id}/{run_name}')

# Initialize the storage
storage = Storage(
    obs=jnp.zeros((num_steps, num_envs) + envs.single_observation_space.shape),
    actions=jnp.zeros((num_steps, num_envs) + envs.single_action_space.shape),
    logprobs=jnp.zeros((num_steps, num_envs)),
    dones=jnp.zeros((num_steps, num_envs)),
    values=jnp.zeros((num_steps, num_envs)),
    advantages=jnp.zeros((num_steps, num_envs)),
    returns=jnp.zeros((num_steps, num_envs)),
    rewards=jnp.zeros((num_steps, num_envs)),
)
global_step = 0
start_time = time.time()
next_obs, _ = envs.reset(seed=seed)
next_done = jnp.zeros(num_envs)

try:
    for update in tqdm(range(1, num_updates + 1)):
        next_obs, next_done, storage, action_key, global_step = rollout(agent_state, next_obs, next_done, storage, action_key, global_step, writer)
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, explained_var, permutation_key = update_ppo(agent_state, storage, permutation_key)

        writer.scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
        writer.scalar("losses/value_loss", v_loss.item(), global_step)
        writer.scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.scalar("losses/explained_variance", explained_var, global_step)
        writer.scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    print(colored('Training complete!', 'green'))
except KeyboardInterrupt:
    print(colored('Training interrupted!', 'red'))
finally:
    envs.close()
    writer.close()
    wandb.finish()