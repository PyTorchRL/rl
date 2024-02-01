# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import _pickle
import abc

import contextlib

import functools

import inspect
import os
import queue
import sys
import time
import warnings
from collections import OrderedDict
from copy import deepcopy
from multiprocessing import connection, queues
from multiprocessing.managers import SyncManager
from textwrap import indent
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    TensorDict,
    TensorDictBase,
    TensorDictParams,
)
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from tensordict.utils import NestedKey
from torch import multiprocessing as mp
from torch.utils._pytree import tree_map
from torch.utils.data import IterableDataset

from torchrl._utils import (
    _check_for_faulty_process,
    _ProcessNoWarn,
    accept_remote_rref_udf_invocation,
    logger as torchrl_logger,
    prod,
    RL_WARNINGS,
    VERBOSE,
)
from torchrl.collectors.utils import split_trajectories
from torchrl.data.tensor_specs import CompositeSpec, TensorSpec
from torchrl.data.utils import CloudpickleWrapper, DEVICE_TYPING
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms import StepCounter, TransformedEnv
from torchrl.envs.utils import (
    _aggregate_end_of_traj,
    _convert_exploration_type,
    ExplorationType,
    set_exploration_type,
)

_TIMEOUT = 1.0
_MIN_TIMEOUT = 1e-3  # should be several orders of magnitude inferior wrt time spent collecting a trajectory
# MAX_IDLE_COUNT is the maximum number of times a Dataloader worker can timeout with his queue.
_MAX_IDLE_COUNT = int(os.environ.get("MAX_IDLE_COUNT", 1000))

DEFAULT_EXPLORATION_TYPE: ExplorationType = ExplorationType.RANDOM

_is_osx = sys.platform.startswith("darwin")


class RandomPolicy:
    """A random policy for data collectors.

    This is a wrapper around the action_spec.rand method.

    Args:
        action_spec: TensorSpec object describing the action specs

    Examples:
        >>> from tensordict import TensorDict
        >>> from torchrl.data.tensor_specs import BoundedTensorSpec
        >>> action_spec = BoundedTensorSpec(-torch.ones(3), torch.ones(3))
        >>> actor = RandomPolicy(action_spec=action_spec)
        >>> td = actor(TensorDict({}, batch_size=[])) # selects a random action in the cube [-1; 1]
    """

    def __init__(self, action_spec: TensorSpec, action_key: NestedKey = "action"):
        super().__init__()
        self.action_spec = action_spec.clone()
        self.action_key = action_key

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        if isinstance(self.action_spec, CompositeSpec):
            return td.update(self.action_spec.rand())
        else:
            return td.set(self.action_key, self.action_spec.rand())


class _Interruptor:
    """A class for managing the collection state of a process.

    This class provides methods to start and stop collection, and to check
    whether collection has been stopped. The collection state is protected
    by a lock to ensure thread-safety.
    """

    # interrupter vs interruptor: google trends seems to indicate that "or" is more
    # widely used than "er" even if my IDE complains about that...
    def __init__(self):
        self._collect = True
        self._lock = mp.Lock()

    def start_collection(self):
        with self._lock:
            self._collect = True

    def stop_collection(self):
        with self._lock:
            self._collect = False

    def collection_stopped(self):
        with self._lock:
            return self._collect is False


class _InterruptorManager(SyncManager):
    """A custom SyncManager for managing the collection state of a process.

    This class extends the SyncManager class and allows to share an Interruptor object
    between processes.
    """

    pass


_InterruptorManager.register("_Interruptor", _Interruptor)


def recursive_map_to_cpu(dictionary: OrderedDict) -> OrderedDict:
    """Maps the tensors to CPU through a nested dictionary."""
    return OrderedDict(
        **{
            k: recursive_map_to_cpu(item)
            if isinstance(item, OrderedDict)
            else item.cpu()
            if isinstance(item, torch.Tensor)
            else item
            for k, item in dictionary.items()
        }
    )


def _policy_is_tensordict_compatible(policy: nn.Module):
    if isinstance(policy, _NonParametricPolicyWrapper) and isinstance(
            policy.policy, RandomPolicy
    ):
        return True

    if isinstance(policy, TensorDictModuleBase):
        return True

    sig = inspect.signature(policy.forward)

    if (
            len(sig.parameters) == 1
            and hasattr(policy, "in_keys")
            and hasattr(policy, "out_keys")
    ):
        raise RuntimeError(
            "Passing a policy that is not a tensordict.nn.TensorDictModuleBase subclass but has in_keys and out_keys "
            "is deprecated. Users should inherit from this class (which "
            "has very few restrictions) to make the experience smoother. "
            "Simply change your policy from `class Policy(nn.Module)` to `Policy(tensordict.nn.TensorDictModuleBase)` "
            "and this error should disappear.",
        )
    elif not hasattr(policy, "in_keys") and not hasattr(policy, "out_keys"):
        # if it's not a TensorDictModule, and in_keys and out_keys are not defined then
        # we assume no TensorDict compatibility and will try to wrap it.
        return False

    # if in_keys or out_keys were defined but policy is not a TensorDictModule or
    # accepts multiple arguments then it's likely the user is trying to do something
    # that will have undetermined behaviour, we raise an error
    raise TypeError(
        "Received a policy that defines in_keys or out_keys and also expects multiple "
        "arguments to policy.forward. If the policy is compatible with TensorDict, it "
        "should take a single argument of type TensorDict to policy.forward and define "
        "both in_keys and out_keys. Alternatively, policy.forward can accept "
        "arbitrarily many tensor inputs and leave in_keys and out_keys undefined and "
        "TorchRL will attempt to automatically wrap the policy with a TensorDictModule."
    )


class DataCollectorBase(IterableDataset, metaclass=abc.ABCMeta):
    """Base class for data collectors."""

    _iterator = None

    def _make_compatible_policy(self, policy, observation_spec=None):
        if policy is None:
            if not hasattr(self, "env") or self.env is None:
                raise ValueError(
                    "env must be provided to _get_policy_and_device if policy is None"
                )
            policy = RandomPolicy(self.env.input_spec["full_action_spec"])
        # make sure policy is an nn.Module
        policy = _NonParametricPolicyWrapper(policy)
        if not _policy_is_tensordict_compatible(policy):
            # policy is a nn.Module that doesn't operate on tensordicts directly
            # so we attempt to auto-wrap policy with TensorDictModule
            if observation_spec is None:
                raise ValueError(
                    "Unable to read observation_spec from the environment. This is "
                    "required to check compatibility of the environment and policy "
                    "since the policy is a nn.Module that operates on tensors "
                    "rather than a TensorDictModule or a nn.Module that accepts a "
                    "TensorDict as input and defines in_keys and out_keys."
                )

            try:
                # signature modified by make_functional
                sig = policy.forward.__signature__
            except AttributeError:
                sig = inspect.signature(policy.forward)
            required_kwargs = {
                str(k) for k, p in sig.parameters.items() if p.default is inspect._empty
            }
            next_observation = {
                key: value for key, value in observation_spec.rand().items()
            }
            # we check if all the mandatory params are there
            params = list(sig.parameters.keys())
            if (
                    set(sig.parameters) == {"tensordict"}
                    or set(sig.parameters) == {"td"}
                    or (
                    len(params) == 1
                    and is_tensor_collection(sig.parameters[params[0]].annotation)
            )
            ):
                pass
            elif not required_kwargs.difference(set(next_observation)):
                in_keys = [str(k) for k in sig.parameters if k in next_observation]
                if not hasattr(self, "env") or self.env is None:
                    out_keys = ["action"]
                else:
                    out_keys = list(self.env.action_keys)
                for p in policy.parameters():
                    policy_device = p.device
                    break
                else:
                    policy_device = None
                if policy_device:
                    next_observation = tree_map(
                        lambda x: x.to(policy_device), next_observation
                    )
                output = policy(**next_observation)

                if isinstance(output, tuple):
                    out_keys.extend(f"output{i + 1}" for i in range(len(output) - 1))

                policy = TensorDictModule(policy, in_keys=in_keys, out_keys=out_keys)
            else:
                raise TypeError(
                    f"""Arguments to policy.forward are incompatible with entries in
env.observation_spec (got incongruent signatures: fun signature is {set(sig.parameters)} vs specs {set(next_observation)}).
If you want TorchRL to automatically wrap your policy with a TensorDictModule
then the arguments to policy.forward must correspond one-to-one with entries
in env.observation_spec that are prefixed with 'next_'. For more complex
behaviour and more control you can consider writing your own TensorDictModule.
Check the collector documentation to know more about accepted policies.
"""
                )
        return policy

    def _get_policy_and_device(
            self,
            policy: Optional[
                Union[
                    TensorDictModule,
                    Callable[[TensorDictBase], TensorDictBase],
                ]
            ] = None,
            observation_spec: TensorSpec = None,
    ) -> Tuple[TensorDictModule, Union[None, Callable[[], dict]]]:
        """Util method to get a policy and its device given the collector __init__ inputs.

        Args:
            create_env_fn (Callable or list of callables): an env creator
                function (or a list of creators)
            create_env_kwargs (dictionary): kwargs for the env creator
            policy (TensorDictModule, optional): a policy to be used
            observation_spec (TensorSpec, optional): spec of the observations

        """
        policy = self._make_compatible_policy(policy, observation_spec)
        param_and_buf = TensorDict.from_module(policy, as_module=True)

        def get_weights_fn(param_and_buf=param_and_buf):
            return param_and_buf.data

        if self.policy_device:
            # create a stateless policy and populate it with params
            def _map_to_device_params(param, device):
                is_param = isinstance(param, nn.Parameter)

                pd = param.detach().to(device, non_blocking=True)

                if is_param:
                    pd = nn.Parameter(pd, requires_grad=False)
                return pd

            # Create a stateless policy, then populate this copy with params on device
            with param_and_buf.apply(
                    functools.partial(_map_to_device_params, device="meta")
            ).to_module(policy):
                policy = deepcopy(policy)

            param_and_buf.apply(
                functools.partial(_map_to_device_params, device=self.policy_device)
            ).to_module(policy)

        return policy, get_weights_fn

    def update_policy_weights_(
            self, policy_weights: Optional[TensorDictBase] = None
    ) -> None:
        """Updates the policy weights if the policy of the data collector and the trained policy live on different devices.

        Args:
            policy_weights (TensorDictBase, optional): if provided, a TensorDict containing
                the weights of the policy to be used for the udpdate.

        """
        if policy_weights is not None:
            self.policy_weights.data.update_(policy_weights)
        elif self.get_weights_fn is not None:
            self.policy_weights.data.update_(self.get_weights_fn())

    def __iter__(self) -> Iterator[TensorDictBase]:
        return self.iterator()

    def next(self):
        try:
            if self._iterator is None:
                self._iterator = iter(self)
            out = next(self._iterator)
            # if any, we don't want the device ref to be passed in distributed settings
            out.clear_device_()
            return out
        except StopIteration:
            return None

    @abc.abstractmethod
    def shutdown(self):
        raise NotImplementedError

    @abc.abstractmethod
    def iterator(self) -> Iterator[TensorDictBase]:
        raise NotImplementedError

    @abc.abstractmethod
    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    @abc.abstractmethod
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}()"
        return string


@accept_remote_rref_udf_invocation
class SyncDataCollector(DataCollectorBase):
    """Generic data collector for RL problems. Requires an environment constructor and a policy.

    Args:
        create_env_fn (Callable): a callable that returns an instance of
            :class:`~torchrl.envs.EnvBase` class.
        policy (Callable): Policy to be executed in the environment.
            Must accept :class:`tensordict.tensordict.TensorDictBase` object as input.
            If ``None`` is provided, the policy used will be a
            :class:`~torchrl.collectors.RandomPolicy` instance with the environment
            ``action_spec``.
            Accepted policies are usually subclasses of :class:`~tensordict.nn.TensorDictModuleBase`.
            This is the recommended usage of the collector.
            Other callables are accepted too:
            If the policy is not a ``TensorDictModuleBase`` (e.g., a regular :class:`~torch.nn.Module`
            instances) it will be wrapped in a `nn.Module` first.
            Then, the collector will try to assess if these
            modules require wrapping in a :class:`~tensordict.nn.TensorDictModule` or not.
            - If the policy forward signature matches any of ``forward(self, tensordict)``,
              ``forward(self, td)`` or ``forward(self, <anything>: TensorDictBase)`` (or
              any typing with a single argument typed as a subclass of ``TensorDictBase``)
              then the policy won't be wrapped in a :class:`~tensordict.nn.TensorDictModule`.
            - In all other cases an attempt to wrap it will be undergone as such: ``TensorDictModule(policy, in_keys=env_obs_key, out_keys=env.action_keys)``.

    Keyword Args:
        frames_per_batch (int): A keyword-only argument representing the total
            number of elements in a batch.
        total_frames (int): A keyword-only argument representing the total
            number of frames returned by the collector
            during its lifespan. If the ``total_frames`` is not divisible by
            ``frames_per_batch``, an exception is raised.
             Endless collectors can be created by passing ``total_frames=-1``.
             Defaults to ``-1`` (endless collector).
        device (int, str or torch.device, optional): The generic device of the
            collector. The ``device`` args fills any non-specified device: if
            ``device`` is not ``None`` and any of ``storing_device``, ``policy_device`` or
            ``env_device`` is not specified, its value will be set to ``device``.
            Defaults to ``None`` (No default device).
        storing_device (int, str or torch.device, optional): The device on which
            the output :class:`~tensordict.TensorDict` will be stored.
            If ``device`` is passed and ``storing_device`` is ``None``, it will
            default to the value indicated by ``device``.
            For long trajectories, it may be necessary to store the data on a different
            device than the one where the policy and env are executed.
            Defaults to ``None`` (the output tensordict isn't on a specific device,
            leaf tensors sit on the device where they were created).
        env_device (int, str or torch.device, optional): The device on which
            the environment should be cast (or executed if that functionality is
            supported). If not specified and the env has a non-``None`` device,
            ``env_device`` will default to that value. If ``device`` is passed
            and ``env_device=None``, it will default to ``device``. If the value
            as such specified of ``env_device`` differs from ``policy_device``
            and one of them is not ``None``, the data will be cast to ``env_device``
            before being passed to the env (i.e., passing different devices to
            policy and env is supported). Defaults to ``None``.
        policy_device (int, str or torch.device, optional): The device on which
            the policy should be cast.
            If ``device`` is passed and ``policy_device=None``, it will default
            to ``device``. If the value as such specified of ``policy_device``
            differs from ``env_device`` and one of them is not ``None``,
            the data will be cast to ``policy_device`` before being passed to
            the policy (i.e., passing different devices to policy and env is
            supported). Defaults to ``None``.
        create_env_kwargs (dict, optional): Dictionary of kwargs for
            ``create_env_fn``.
        max_frames_per_traj (int, optional): Maximum steps per trajectory.
            Note that a trajectory can span across multiple batches (unless
            ``reset_at_each_iter`` is set to ``True``, see below).
            Once a trajectory reaches ``n_steps``, the environment is reset.
            If the environment wraps multiple environments together, the number
            of steps is tracked for each environment independently. Negative
            values are allowed, in which case this argument is ignored.
            Defaults to ``None`` (i.e., no maximum number of steps).
        init_random_frames (int, optional): Number of frames for which the
            policy is ignored before it is called. This feature is mainly
            intended to be used in offline/model-based settings, where a
            batch of random trajectories can be used to initialize training.
            If provided, it will be rounded up to the closest multiple of frames_per_batch.
            Defaults to ``None`` (i.e. no random frames).
        reset_at_each_iter (bool, optional): Whether environments should be reset
            at the beginning of a batch collection.
            Defaults to ``False``.
        postproc (Callable, optional): A post-processing transform, such as
            a :class:`~torchrl.envs.Transform` or a :class:`~torchrl.data.postprocs.MultiStep`
            instance.
            Defaults to ``None``.
        split_trajs (bool, optional): Boolean indicating whether the resulting
            TensorDict should be split according to the trajectories.
            See :func:`~torchrl.collectors.utils.split_trajectories` for more
            information.
            Defaults to ``False``.
        exploration_type (ExplorationType, optional): interaction mode to be used when
            collecting data. Must be one of ``torchrl.envs.utils.ExplorationType.RANDOM``,
            ``torchrl.envs.utils.ExplorationType.MODE`` or ``torchrl.envs.utils.ExplorationType.MEAN``.
            Defaults to ``torchrl.envs.utils.ExplorationType.RANDOM``.
        return_same_td (bool, optional): if ``True``, the same TensorDict
            will be returned at each iteration, with its values
            updated. This feature should be used cautiously: if the same
            tensordict is added to a replay buffer for instance,
            the whole content of the buffer will be identical.
            Default is ``False``.
        interruptor (_Interruptor, optional):
            An _Interruptor object that can be used from outside the class to control rollout collection.
            The _Interruptor class has methods ´start_collection´ and ´stop_collection´, which allow to implement
            strategies such as preeptively stopping rollout collection.
            Default is ``False``.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from tensordict.nn import TensorDictModule
        >>> from torch import nn
        >>> env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
        >>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
        >>> collector = SyncDataCollector(
        ...     create_env_fn=env_maker,
        ...     policy=policy,
        ...     total_frames=2000,
        ...     max_frames_per_traj=50,
        ...     frames_per_batch=200,
        ...     init_random_frames=-1,
        ...     reset_at_each_iter=False,
        ...     device="cpu",
        ...     storing_device="cpu",
        ... )
        >>> for i, data in enumerate(collector):
        ...     if i == 2:
        ...         print(data)
        ...         break
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                collector: TensorDict(
                    fields={
                        traj_ids: Tensor(shape=torch.Size([200]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        step_count: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                        truncated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                step_count: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                truncated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([200]),
            device=cpu,
            is_shared=False)
        >>> del collector

    The collector delivers batches of data that are marked with a ``"time"``
    dimension.

    Examples:
        >>> assert data.names[-1] == "time"

    """

    def __init__(
            self,
            create_env_fn: Union[
                EnvBase, "EnvCreator", Sequence[Callable[[], EnvBase]]  # noqa: F821
            ],  # noqa: F821
            policy: Optional[
                Union[
                    TensorDictModule,
                    Callable[[TensorDictBase], TensorDictBase],
                ]
            ],
            *,
            frames_per_batch: int,
            total_frames: int = -1,
            device: DEVICE_TYPING = None,
            storing_device: DEVICE_TYPING = None,
            policy_device: DEVICE_TYPING = None,
            env_device: DEVICE_TYPING = None,
            create_env_kwargs: dict | None = None,
            max_frames_per_traj: int | None = None,
            init_random_frames: int | None = None,
            reset_at_each_iter: bool = False,
            postproc: Callable[[TensorDictBase], TensorDictBase] | None = None,
            split_trajs: bool | None = None,
            exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
            exploration_mode=None,
            return_same_td: bool = False,
            reset_when_done: bool = True,
            interruptor=None,
    ):
        from torchrl.envs.batched_envs import _BatchedEnv

        self.closed = True

        exploration_type = _convert_exploration_type(
            exploration_mode=exploration_mode, exploration_type=exploration_type
        )
        if create_env_kwargs is None:
            create_env_kwargs = {}
        if not isinstance(create_env_fn, EnvBase):
            env = create_env_fn(**create_env_kwargs)
        else:
            env = create_env_fn
            if create_env_kwargs:
                if not isinstance(env, _BatchedEnv):
                    raise RuntimeError(
                        "kwargs were passed to SyncDataCollector but they can't be set "
                        f"on environment of type {type(create_env_fn)}."
                    )
                env.update_kwargs(create_env_kwargs)

        ##########################
        # Setting devices:
        # The rule is the following:
        # - If no device is passed, all devices are assumed to work OOB.
        #   The tensordict used for output is not on any device (ie, actions and observations
        #   can be on a different device).
        # - If the ``device`` is passed, it is used for all devices (storing, env and policy)
        #   unless overridden by another kwarg.
        # - The rest of the kwargs control the respective device.
        storing_device, policy_device, env_device = self._get_devices(
            storing_device=storing_device,
            policy_device=policy_device,
            env_device=env_device,
            device=device,
        )

        self.storing_device = storing_device
        self.env_device = env_device
        self.policy_device = policy_device
        self.device = device
        # Check if we need to cast things from device to device
        # If the policy has a None device and the env too, no need to cast (we don't know
        # and assume the user knows what she's doing).
        # If the devices match we're happy too.
        # Only if the values differ we need to cast
        self._cast_to_policy_device = self.policy_device != self.env_device

        self.env: EnvBase = env
        del env
        self.closed = False
        if not reset_when_done:
            raise ValueError("reset_when_done is deprectated.")
        self.reset_when_done = reset_when_done
        self.n_env = self.env.batch_size.numel()

        (self.policy, self.get_weights_fn,) = self._get_policy_and_device(
            policy=policy,
            observation_spec=self.env.observation_spec,
        )

        if isinstance(self.policy, nn.Module):
            self.policy_weights = TensorDict.from_module(self.policy, as_module=True)
        else:
            self.policy_weights = TensorDict({}, [])

        if self.env_device:
            self.env: EnvBase = self.env.to(self.env_device)
        elif self.env.device is not None:
            # we we did not receive an env device, we use the device of the env
            self.env_device = self.env.device

        self.max_frames_per_traj = (
            int(max_frames_per_traj) if max_frames_per_traj is not None else 0
        )
        if self.max_frames_per_traj is not None and self.max_frames_per_traj > 0:
            # let's check that there is no StepCounter yet
            for key in self.env.output_spec.keys(True, True):
                if isinstance(key, str):
                    key = (key,)
                if "step_count" in key:
                    raise ValueError(
                        "A 'step_count' key is already present in the environment "
                        "and the 'max_frames_per_traj' argument may conflict with "
                        "a 'StepCounter' that has already been set. "
                        "Possible solutions: Set max_frames_per_traj to 0 or "
                        "remove the StepCounter limit from the environment transforms."
                    )
            self.env = TransformedEnv(
                self.env, StepCounter(max_steps=self.max_frames_per_traj)
            )

        if total_frames is None or total_frames < 0:
            total_frames = float("inf")
        else:
            remainder = total_frames % frames_per_batch
            if remainder != 0 and RL_WARNINGS:
                warnings.warn(
                    f"total_frames ({total_frames}) is not exactly divisible by frames_per_batch ({frames_per_batch})."
                    f"This means {frames_per_batch - remainder} additional frames will be collected."
                    "To silence this message, set the environment variable RL_WARNINGS to False."
                )
        self.total_frames = (
            int(total_frames) if total_frames != float("inf") else total_frames
        )
        self.reset_at_each_iter = reset_at_each_iter
        self.init_random_frames = (
            int(init_random_frames) if init_random_frames is not None else 0
        )
        if (
                init_random_frames is not None
                and init_random_frames % frames_per_batch != 0
                and RL_WARNINGS
        ):
            warnings.warn(
                f"init_random_frames ({init_random_frames}) is not exactly a multiple of frames_per_batch ({frames_per_batch}), "
                f" this results in more init_random_frames than requested"
                f" ({-(-init_random_frames // frames_per_batch) * frames_per_batch})."
                "To silence this message, set the environment variable RL_WARNINGS to False."
            )

        self.postproc = postproc
        if (
                self.postproc is not None
                and hasattr(self.postproc, "to")
                and self.storing_device
        ):
            self.postproc.to(self.storing_device)
        if frames_per_batch % self.n_env != 0 and RL_WARNINGS:
            warnings.warn(
                f"frames_per_batch ({frames_per_batch}) is not exactly divisible by the number of batched environments ({self.n_env}), "
                f" this results in more frames_per_batch per iteration that requested"
                f" ({-(-frames_per_batch // self.n_env) * self.n_env})."
                "To silence this message, set the environment variable RL_WARNINGS to False."
            )
        self.requested_frames_per_batch = int(frames_per_batch)
        self.frames_per_batch = -(-frames_per_batch // self.n_env)
        self.exploration_type = (
            exploration_type if exploration_type else DEFAULT_EXPLORATION_TYPE
        )
        self.return_same_td = return_same_td

        # Shuttle is a deviceless tensordict that just carried data from env to policy and policy to env
        self._shuttle = self.env.reset()
        if self.policy_device != self.env_device or self.env_device is None:
            self._shuttle_has_no_device = True
            self._shuttle.clear_device_()
        else:
            self._shuttle_has_no_device = False

        traj_ids = torch.arange(self.n_env, device=self.storing_device).view(
            self.env.batch_size
        )
        self._shuttle.set(
            ("collector", "traj_ids"),
            traj_ids,
        )
        with torch.no_grad():
            self._final_rollout = self.env.fake_tensordict()

        # If storing device is not None, we use this to cast the storage.
        # If it is None and the env and policy are on the same device,
        # the storing device is already the same as those, so we don't need
        # to consider this use case.
        # In all other cases, we can't really put a device on the storage,
        # since at least one data source has a device that is not clear.
        if self.storing_device:
            self._final_rollout = self._final_rollout.to(
                self.storing_device, non_blocking=True
            )
        else:
            # erase all devices
            self._final_rollout.clear_device_()

        # If the policy has a valid spec, we use it
        self._policy_output_keys = set()
        if (
                hasattr(self.policy, "spec")
                and self.policy.spec is not None
                and all(v is not None for v in self.policy.spec.values(True, True))
        ):
            if any(
                    key not in self._final_rollout.keys(isinstance(key, tuple))
                    for key in self.policy.spec.keys(True, True)
            ):
                # if policy spec is non-empty, all the values are not None and the keys
                # match the out_keys we assume the user has given all relevant information
                # the policy could have more keys than the env:
                policy_spec = self.policy.spec
                if policy_spec.ndim < self._final_rollout.ndim:
                    policy_spec = policy_spec.expand(self._final_rollout.shape)
                for key, spec in policy_spec.items(True, True):
                    self._policy_output_keys.add(key)
                    if key in self._final_rollout.keys(True):
                        continue
                    self._final_rollout.set(key, spec.zero())

        else:
            # otherwise, we perform a small number of steps with the policy to
            # determine the relevant keys with which to pre-populate _final_rollout.
            # This is the safest thing to do if the spec has None fields or if there is
            # no spec at all.
            # See #505 for additional context.
            self._final_rollout.update(self._shuttle.copy())
            with torch.no_grad():
                policy_input = self._shuttle.copy()
                if self.policy_device:
                    policy_input = policy_input.to(self.policy_device)
                # we cast to policy device, we'll deal with the device later
                policy_input_copy = policy_input.copy()
                policy_input_clone = (
                    policy_input.clone()
                )  # to test if values have changed in-place
                policy_output = self.policy(policy_input)

                # check that we don't have exclusive keys, because they don't appear in keys
                def check_exclusive(val):
                    if (
                            isinstance(val, LazyStackedTensorDict)
                            and val._has_exclusive_keys
                    ):
                        raise RuntimeError(
                            "LazyStackedTensorDict with exclusive keys are not permitted in collectors. "
                            "Consider using a placeholder for missing keys."
                        )

                policy_output._fast_apply(check_exclusive, call_on_nested=True)
                # Use apply, because it works well with lazy stacks
                # Edge-case of this approach: the policy may change the values in-place and only by a tiny bit
                # or occasionally. In these cases, the keys will be missed (we can't detect if the policy has
                # changed them here).
                # This will cause a failure to update entries when policy and env device mismatch and
                # casting is necessary.
                filtered_policy_output = policy_output.apply(
                    lambda value_output, value_input, value_input_clone: value_output
                    if (value_input is None)
                       or (value_output is not value_input)
                       or ~torch.isclose(value_output, value_input_clone).any()
                    else None,
                    policy_input_copy,
                    policy_input_clone,
                    default=None,
                )
                self._policy_output_keys = list(
                    self._policy_output_keys.union(
                        set(filtered_policy_output.keys(True, True))
                    )
                )
                self._final_rollout.update(
                    policy_output.select(*self._policy_output_keys)
                )
                del filtered_policy_output, policy_output, policy_input

        _env_output_keys = []
        for spec in ["full_observation_spec", "full_done_spec", "full_reward_spec"]:
            _env_output_keys += list(self.env.output_spec[spec].keys(True, True))
        self._env_output_keys = _env_output_keys
        self._final_rollout = (
            self._final_rollout.unsqueeze(-1)
            .expand(*self.env.batch_size, self.frames_per_batch)
            .clone()
            .zero_()
        )

        # in addition to outputs of the policy, we add traj_ids to
        # _final_rollout which will be collected during rollout
        self._final_rollout.set(
            ("collector", "traj_ids"),
            torch.zeros(
                *self._final_rollout.batch_size,
                dtype=torch.int64,
                device=self.storing_device,
            ),
        )
        self._final_rollout.refine_names(..., "time")

        if split_trajs is None:
            split_trajs = False
        self.split_trajs = split_trajs
        self._exclude_private_keys = True
        self.interruptor = interruptor
        self._frames = 0
        self._iter = -1

    @classmethod
    def _get_devices(
            cls,
            *,
            storing_device: torch.device,
            policy_device: torch.device,
            env_device: torch.device,
            device: torch.device,
    ):
        device = torch.device(device) if device else device
        storing_device = torch.device(storing_device) if storing_device else device
        policy_device = torch.device(policy_device) if policy_device else device
        env_device = torch.device(env_device) if env_device else device
        if storing_device is None and (env_device == policy_device):
            storing_device = env_device
        return storing_device, policy_device, env_device

    # for RPC
    def next(self):
        return super().next()

    # for RPC
    def update_policy_weights_(
            self, policy_weights: Optional[TensorDictBase] = None
    ) -> None:
        super().update_policy_weights_(policy_weights)

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        """Sets the seeds of the environments stored in the DataCollector.

        Args:
            seed (int): integer representing the seed to be used for the environment.
            static_seed(bool, optional): if ``True``, the seed is not incremented.
                Defaults to False

        Returns:
            Output seed. This is useful when more than one environment is contained in the DataCollector, as the
            seed will be incremented for each of these. The resulting seed is the seed of the last environment.

        Examples:
            >>> from torchrl.envs import ParallelEnv
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> from tensordict.nn import TensorDictModule
            >>> from torch import nn
            >>> env_fn = lambda: GymEnv("Pendulum-v1")
            >>> env_fn_parallel = ParallelEnv(6, env_fn)
            >>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
            >>> collector = SyncDataCollector(env_fn_parallel, policy, total_frames=300, frames_per_batch=100)
            >>> out_seed = collector.set_seed(1)  # out_seed = 6

        """
        return self.env.set_seed(seed, static_seed=static_seed)

    def iterator(self) -> Iterator[TensorDictBase]:
        """Iterates through the DataCollector.

        Yields: TensorDictBase objects containing (chunks of) trajectories

        """
        if self.storing_device and self.storing_device.type == "cuda":
            stream = torch.cuda.Stream(self.storing_device, priority=-1)
            event = stream.record_event()
            streams = [stream]
            events = [event]
        elif self.storing_device is None:
            streams = []
            events = []
            # this way of checking cuda is robust to lazy stacks with mismatching shapes
            cuda_devices = set()

            def cuda_check(tensor: torch.Tensor):
                if tensor.is_cuda:
                    cuda_devices.add(tensor.device)

            self._final_rollout.apply(cuda_check)
            for device in cuda_devices:
                streams.append(torch.cuda.Stream(device, priority=-1))
                events.append(streams[-1].record_event())
        else:
            streams = []
            events = []
        with contextlib.ExitStack() as stack:
            for stream in streams:
                stack.enter_context(torch.cuda.stream(stream))

            total_frames = self.total_frames

            while self._frames < self.total_frames:
                self._iter += 1
                tensordict_out = self.rollout()
                self._frames += tensordict_out.numel()
                if self._frames >= total_frames:
                    self.env.close()

                if self.split_trajs:
                    tensordict_out = split_trajectories(
                        tensordict_out, prefix="collector"
                    )
                if self.postproc is not None:
                    tensordict_out = self.postproc(tensordict_out)
                if self._exclude_private_keys:

                    def is_private(key):
                        if isinstance(key, str) and key.startswith("_"):
                            return True
                        if isinstance(key, tuple) and any(
                                _key.startswith("_") for _key in key
                        ):
                            return True
                        return False

                    excluded_keys = [
                        key for key in tensordict_out.keys(True) if is_private(key)
                    ]
                    tensordict_out = tensordict_out.exclude(
                        *excluded_keys, inplace=True
                    )
                if self.return_same_td:
                    # This is used with multiprocessed collectors to use the buffers
                    # stored in the tensordict.
                    if events:
                        for event in events:
                            event.record()
                            event.synchronize()
                    yield tensordict_out
                else:
                    # we must clone the values, as the tensordict is updated in-place.
                    # otherwise the following code may break:
                    # >>> for i, data in enumerate(collector):
                    # >>>      if i == 0:
                    # >>>          data0 = data
                    # >>>      elif i == 1:
                    # >>>          data1 = data
                    # >>>      else:
                    # >>>          break
                    # >>> assert data0["done"] is not data1["done"]
                    yield tensordict_out.clone()

    def _update_traj_ids(self, env_output) -> None:
        # we can't use the reset keys because they're gone
        traj_sop = _aggregate_end_of_traj(
            env_output.get("next"), done_keys=self.env.done_keys
        )
        if traj_sop.any():
            traj_ids = self._shuttle.get(("collector", "traj_ids"))
            traj_sop = traj_sop.to(self.storing_device)
            traj_ids = traj_ids.clone().to(self.storing_device)
            traj_ids[traj_sop] = traj_ids.max() + torch.arange(
                1,
                traj_sop.sum() + 1,
                device=self.storing_device,
                )
            self._shuttle.set(("collector", "traj_ids"), traj_ids)

    @torch.no_grad()
    def rollout(self) -> TensorDictBase:
        """Computes a rollout in the environment using the provided policy.

        Returns:
            TensorDictBase containing the computed rollout.

        """
        if self.reset_at_each_iter:
            self._shuttle.update(self.env.reset())

        # self._shuttle.fill_(("collector", "step_count"), 0)
        self._final_rollout.fill_(("collector", "traj_ids"), -1)
        tensordicts = []
        with set_exploration_type(self.exploration_type):
            for t in range(self.frames_per_batch):
                if (
                        self.init_random_frames is not None
                        and self._frames < self.init_random_frames
                ):
                    self.env.rand_action(self._shuttle)
                else:
                    if self._cast_to_policy_device:
                        if self.policy_device is not None:
                            policy_input = self._shuttle.to(
                                self.policy_device, non_blocking=True
                            )
                        elif self.policy_device is None:
                            # we know the tensordict has a device otherwise we would not be here
                            # we can pass this, clear_device_ must have been called earlier
                            # policy_input = self._shuttle.clear_device_()
                            policy_input = self._shuttle
                    else:
                        policy_input = self._shuttle
                    # we still do the assignment for security
                    policy_output = self.policy(policy_input)
                    if self._shuttle is not policy_output:
                        # ad-hoc update shuttle
                        self._shuttle.update(
                            policy_output, keys_to_update=self._policy_output_keys
                        )

                if self._cast_to_policy_device:
                    if self.env_device is not None:
                        env_input = self._shuttle.to(self.env_device, non_blocking=True)
                    elif self.env_device is None:
                        # we know the tensordict has a device otherwise we would not be here
                        # we can pass this, clear_device_ must have been called earlier
                        # env_input = self._shuttle.clear_device_()
                        env_input = self._shuttle
                else:
                    env_input = self._shuttle
                env_output, env_next_output = self.env.step_and_maybe_reset(env_input)

                if self._shuttle is not env_output:
                    # ad-hoc update shuttle
                    next_data = env_output.get("next")
                    if self._shuttle_has_no_device:
                        # Make sure
                        next_data.clear_device_()
                    self._shuttle.set("next", next_data)

                if self.storing_device is not None:
                    tensordicts.append(
                        self._shuttle.to(self.storing_device, non_blocking=False)
                    )
                else:
                    tensordicts.append(self._shuttle)

                # carry over collector data without messing up devices
                collector_data = self._shuttle.get("collector").copy()
                self._shuttle = env_next_output
                if self._shuttle_has_no_device:
                    self._shuttle.clear_device_()
                self._shuttle.set("collector", collector_data)

                self._update_traj_ids(env_output)

                if (
                        self.interruptor is not None
                        and self.interruptor.collection_stopped()
                ):
                    try:
                        torch.stack(
                            tensordicts,
                            self._final_rollout.ndim - 1,
                            out=self._final_rollout[: t + 1],
                            )
                    except RuntimeError:
                        with self._final_rollout.unlock_():
                            torch.stack(
                                tensordicts,
                                self._final_rollout.ndim - 1,
                                out=self._final_rollout[: t + 1],
                                )
                    break
            else:
                try:
                    self._final_rollout = torch.stack(
                        tensordicts,
                        self._final_rollout.ndim - 1,
                        out=self._final_rollout,
                        )
                except RuntimeError:
                    with self._final_rollout.unlock_():
                        self._final_rollout = torch.stack(
                            tensordicts,
                            self._final_rollout.ndim - 1,
                            out=self._final_rollout,
                            )
        return self._final_rollout

    @staticmethod
    def _update_device_wise(tensor0, tensor1):
        # given 2 tensors, returns tensor0 if their identity matches,
        # or a copy of tensor1 on the device of tensor0 otherwise
        if tensor1 is None or tensor1 is tensor0:
            return tensor0
        if tensor1.device == tensor0.device:
            return tensor1
        return tensor1.to(tensor0.device, non_blocking=True)

    def reset(self, index=None, **kwargs) -> None:
        """Resets the environments to a new initial state."""
        # metadata
        collector_metadata = self._shuttle.get("collector").clone()
        if index is not None:
            # check that the env supports partial reset
            if prod(self.env.batch_size) == 0:
                raise RuntimeError("resetting unique env with index is not permitted.")
            for reset_key, done_keys in zip(
                    self.env.reset_keys, self.env.done_keys_groups
            ):
                _reset = torch.zeros(
                    self.env.full_done_spec[done_keys[0]].shape,
                    dtype=torch.bool,
                    device=self.env.device,
                )
                _reset[index] = 1
                self._shuttle.set(reset_key, _reset)
        else:
            _reset = None
            self._shuttle.zero_()

        self._shuttle.update(self.env.reset(**kwargs), inplace=True)
        collector_metadata["traj_ids"] = (
                collector_metadata["traj_ids"] - collector_metadata["traj_ids"].min()
        )
        self._shuttle["collector"] = collector_metadata

    def shutdown(self) -> None:
        """Shuts down all workers and/or closes the local environment."""
        if not self.closed:
            self.closed = True
            del self._shuttle, self._final_rollout
            if not self.env.is_closed:
                self.env.close()
            del self.env
        return

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            # an AttributeError will typically be raised if the collector is deleted when the program ends.
            # In the future, insignificant changes to the close method may change the error type.
            # We excplicitely assume that any error raised during closure in
            # __del__ will not affect the program.
            pass

    def state_dict(self) -> OrderedDict:
        """Returns the local state_dict of the data collector (environment and policy).

        Returns:
            an ordered dictionary with fields :obj:`"policy_state_dict"` and
            `"env_state_dict"`.

        """
        from torchrl.envs.batched_envs import _BatchedEnv

        if isinstance(self.env, TransformedEnv):
            env_state_dict = self.env.transform.state_dict()
        elif isinstance(self.env, _BatchedEnv):
            env_state_dict = self.env.state_dict()
        else:
            env_state_dict = OrderedDict()

        if hasattr(self.policy, "state_dict"):
            policy_state_dict = self.policy.state_dict()
            state_dict = OrderedDict(
                policy_state_dict=policy_state_dict,
                env_state_dict=env_state_dict,
            )
        else:
            state_dict = OrderedDict(env_state_dict=env_state_dict)

        state_dict.update({"frames": self._frames, "iter": self._iter})

        return state_dict

    def load_state_dict(self, state_dict: OrderedDict, **kwargs) -> None:
        """Loads a state_dict on the environment and policy.

        Args:
            state_dict (OrderedDict): ordered dictionary containing the fields
                `"policy_state_dict"` and :obj:`"env_state_dict"`.

        """
        strict = kwargs.get("strict", True)
        if strict or "env_state_dict" in state_dict:
            self.env.load_state_dict(state_dict["env_state_dict"], **kwargs)
        if strict or "policy_state_dict" in state_dict:
            self.policy.load_state_dict(state_dict["policy_state_dict"], **kwargs)
        self._frames = state_dict["frames"]
        self._iter = state_dict["iter"]

    def __repr__(self) -> str:
        env_str = indent(f"env={self.env}", 4 * " ")
        policy_str = indent(f"policy={self.policy}", 4 * " ")
        td_out_str = indent(f"td_out={self._final_rollout}", 4 * " ")
        string = (
            f"{self.__class__.__name__}("
            f"\n{env_str},"
            f"\n{policy_str},"
            f"\n{td_out_str},"
            f"\nexploration={self.exploration_type})"
        )
        return string
