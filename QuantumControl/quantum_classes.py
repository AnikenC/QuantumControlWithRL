
from copy import deepcopy
from typing import Any, Callable, Iterator, List, Optional, Sequence, Union, Tuple

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Space
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from gymnasium.vector.utils.spaces import batch_space

class CustomVectorEnv(gym.Env):
    r"""Base class for vectorized environments.

    Each observation returned from vectorized environment is a batch of observations
    for each sub-environment. And :meth:`step` is also expected to receive a batch of
    actions for each sub-environment.

    .. note::

        All sub-environments should share the identical observation and action spaces.
        In other words, a vector of multiple different environments is not supported.

    Parameters
    ----------
    num_envs : int
        Number of environments in the vectorized environment.

    observation_space : :class:`gym.spaces.Space`
        Observation space of a single environment.

    action_space : :class:`gym.spaces.Space`
        Action space of a single environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.is_vector_env = True
        self.observation_space = batch_space(observation_space, n=num_envs)
        self.action_space = batch_space(action_space, n=num_envs)

        self.closed = False
        self.viewer = None

        # The observation and action spaces of a single environment are
        # kept in separate properties
        self.single_observation_space = observation_space
        self.single_action_space = action_space

    def reset_async(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        pass

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        raise NotImplementedError()

    def reset(
        self,
        *,
        seed: Optional[Union[int, List[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        r"""Reset all sub-environments and return a batch of initial observations.

        Returns
        -------
        element of :attr:`observation_space`
            A batch of observations from the vectorized environment.
        """
        self.reset_async(seed=seed, return_info=return_info, options=options)
        return self.reset_wait(seed=seed, return_info=return_info, options=options)

    def step_async(self, actions):
        pass

    def step_wait(self, **kwargs):
        raise NotImplementedError()

    def step(self, actions, simple_sample):
        r"""Take an action for each sub-environments.

        Parameters
        ----------
        actions : element of :attr:`action_space`
            Batch of actions.

        Returns
        -------
        observations : element of :attr:`observation_space`
            A batch of observations from the vectorized environment.

        rewards : :obj:`np.ndarray`, dtype :obj:`np.float_`
            A vector of rewards from the vectorized environment.

        dones : :obj:`np.ndarray`, dtype :obj:`np.bool_`
            A vector whose entries indicate whether the episode has ended.

        infos : list of dict
            A list of auxiliary diagnostic information dicts from sub-environments.
        """

        self.step_async(actions, simple_sample)
        return self.step_wait()

    def call_async(self, name, *args, **kwargs):
        pass

    def call_wait(self, **kwargs):
        raise NotImplementedError()

    def call(self, name, *args, **kwargs):
        """Call a method, or get a property, from each sub-environment.

        Parameters
        ----------
        name : string
            Name of the method or property to call.

        *args
            Arguments to apply to the method call.

        **kwargs
            Keywoard arguments to apply to the method call.

        Returns
        -------
        results : list
            List of the results of the individual calls to the method or
            property for each environment.
        """
        self.call_async(name, *args, **kwargs)
        return self.call_wait()

    def get_attr(self, name):
        """Get a property from each sub-environment.

        Parameters
        ----------
        name : string
            Name of the property to be get from each individual environment.
        """
        return self.call(name)

    def set_attr(self, name, values):
        """Set a property in each sub-environment.

        Parameters
        ----------
        name : string
            Name of the property to be set in each individual environment.

        values : list, tuple, or object
            Values of the property to be set to. If `values` is a list or
            tuple, then it corresponds to the values for each individual
            environment, otherwise a single value is set for all environments.
        """
        raise NotImplementedError()

    def close_extras(self, **kwargs):
        r"""Clean up the extra resources e.g. beyond what's in this base class."""
        pass

    def close(self, **kwargs):
        r"""Close all sub-environments and release resources.

        It also closes all the existing image viewers, then calls :meth:`close_extras` and set
        :attr:`closed` as ``True``.

        .. warning::

            This function itself does not close the environments, it should be handled
            in :meth:`close_extras`. This is generic for both synchronous and asynchronous
            vectorized environments.

        .. note::

            This will be automatically called when garbage collected or program exited.

        """
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras(**kwargs)
        self.closed = True

    def seed(self, seed=None):
        """Set the random seed in all sub-environments.

        Parameters
        ----------
        seed : list of int, or int, optional
            Random seed for each sub-environment. If ``seed`` is a list of
            length ``num_envs``, then the items of the list are chosen as random
            seeds. If ``seed`` is an int, then each sub-environment uses the random
            seed ``seed + n``, where ``n`` is the index of the sub-environment
            (between ``0`` and ``num_envs - 1``).
        """
        deprecation(
            "Function `env.seed(seed)` is marked as deprecated and will be removed in the future. "
            "Please use `env.reset(seed=seed) instead in VectorEnvs."
        )

    def __del__(self):
        if not getattr(self, "closed", True):
            self.close()

    def __repr__(self):
        if self.spec is None:
            return f"{self.__class__.__name__}({self.num_envs})"
        else:
            return f"{self.__class__.__name__}({self.spec.id}, {self.num_envs})"

class CustomSyncVectorEnv(CustomVectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.

    observation_space : :class:`gym.spaces.Space`, optional
        Observation space of a single environment. If ``None``, then the
        observation space of the first environment is taken.

    action_space : :class:`gym.spaces.Space`, optional
        Action space of a single environment. If ``None``, then the action space
        of the first environment is taken.

    copy : bool
        If ``True``, then the :meth:`reset` and :meth:`step` methods return a
        copy of the observations.

    Raises
    ------
    RuntimeError
        If the observation space of some sub-environment does not match
        :obj:`observation_space` (or, by default, the observation space of
        the first sub-environment).

    Example
    -------

    .. code-block::

        >>> env = gym.vector.SyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v0", g=9.81),
        ...     lambda: gym.make("Pendulum-v0", g=1.62)
        ... ])
        >>> env.reset()
        array([[-0.8286432 ,  0.5597771 ,  0.90249056],
               [-0.85009176,  0.5266346 ,  0.60007906]], dtype=float32)
    """

    def __init__(self, env_fns, observation_space=None, action_space=None, copy=True):
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.metadata = self.envs[0].metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super().__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_spaces()
        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

    def seed(self, seed=None):
        super().seed(seed=seed)
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for env, single_seed in zip(self.envs, seed):
            env.seed(single_seed)

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._dones[:] = False
        observations = []
        data_list = []
        for env, single_seed in zip(self.envs, seed):

            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options
            if return_info == True:
                kwargs["return_info"] = return_info

            if not return_info:
                observation = env.reset(**kwargs)
                observations.append(observation)
            else:
                observation, data = env.reset(**kwargs)
                observations.append(observation)
                data_list.append(data)

        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )
        if not return_info:
            return deepcopy(self.observations) if self.copy else self.observations
        else:
            return (
                deepcopy(self.observations) if self.copy else self.observations
            ), data_list

    def step_async(self, actions, simple_sample):
        self._actions = iterate(self.action_space, actions)
        self.simple_sample = simple_sample

    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action, self.simple_sample)
            if self._dones[i]:
                info["terminal_observation"] = observation
                observation = env.reset()
            observations.append(observation)
            infos.append(info)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._dones),
            infos,
        )

    def call(self, name, *args, **kwargs):
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def set_attr(self, name, values):
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            setattr(env, name, value)

    def close_extras(self, **kwargs):
        """Close the environments."""
        [env.close() for env in self.envs]

    def _check_spaces(self):
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError(
                    "Some environments have an observation space different from "
                    f"`{self.single_observation_space}`. In order to batch observations, "
                    "the observation spaces from all environments must be equal."
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.single_action_space}`. In order to batch actions, the "
                    "action spaces from all environments must be equal."
                )

        else:
            return True

class BatchedCustomSyncVectorEnv(CustomVectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Example::

        >>> import gymnasium as gym
        >>> env = gym.vector.SyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v0", g=9.81),
        ...     lambda: gym.make("Pendulum-v0", g=1.62)
        ... ])
        >>> env.reset()
        array([[-0.8286432 ,  0.5597771 ,  0.90249056],
               [-0.85009176,  0.5266346 ,  0.60007906]], dtype=float32)
    """

    def __init__(
        self,
        env_fns: Iterator[Callable[[], Env]],
        num_steps: int = None,
        observation_space: Space = None,
        action_space: Space = None,
        copy: bool = True,
    ):
        """Vectorized environment that serially runs multiple environments.

        Args:
            env_fns: iterable of callable functions that create the environments.
            observation_space: Observation space of a single environment. If ``None``,
                then the observation space of the first environment is taken.
            action_space: Action space of a single environment. If ``None``,
                then the action space of the first environment is taken.
            copy: If ``True``, then the :meth:`reset` and :meth:`step` methods return a copy of the observations.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
        """
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_steps = num_steps
        self.copy = copy
        self.metadata = self.envs[0].metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super().__init__(
            num_envs=len(self.envs),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_spaces()
        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs, self.num_steps), dtype=np.float64)
        self._terminateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

    def seed(self, seed: Optional[Union[int, Sequence[int]]] = None):
        """Sets the seed in all sub-environments.

        Args:
            seed: The seed
        """
        super().seed(seed=seed)
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for env, single_seed in zip(self.envs, seed):
            env.seed(single_seed)

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        Args:
            seed: The reset environment seed
            options: Option information for the environment reset

        Returns:
            The reset observation of the environment and reset information
        """
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._terminateds[:] = False
        self._truncateds[:] = False
        observations = []
        infos = {}
        for i, (env, single_seed) in enumerate(zip(self.envs, seed)):

            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options

            observation, info = env.reset(**kwargs)
            observations.append(observation)
            infos = self._add_info(infos, info, i)

        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )
        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def step_async(self, actions, simple_sample):
        """Sets :attr:`_actions` for use by the :meth:`step_wait` by converting the ``actions`` to an iterable version."""
        self._actions = iterate(self.action_space, actions)
        self.simple_sample = simple_sample

    def step_wait(self):
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):

            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action, self.simple_sample)

            if self._terminateds[i] or self._truncateds[i]:
                old_observation, old_info = observation, info
                observation, info = env.reset()
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )

    def call(self, name, *args, **kwargs) -> tuple:
        """Calls the method with name and applies args and kwargs.

        Args:
            name: The method name
            *args: The method args
            **kwargs: The method kwargs

        Returns:
            Tuple of results
        """
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def set_attr(self, name: str, values: Union[list, tuple, Any]):
        """Sets an attribute of the sub-environments.

        Args:
            name: The property name to change
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise, a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            setattr(env, name, value)

    def close_extras(self, **kwargs):
        """Close the environments."""
        [env.close() for env in self.envs]

    def _check_spaces(self) -> bool:
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError(
                    "Some environments have an observation space different from "
                    f"`{self.single_observation_space}`. In order to batch observations, "
                    "the observation spaces from all environments must be equal."
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.single_action_space}`. In order to batch actions, the "
                    "action spaces from all environments must be equal."
                )

        return True