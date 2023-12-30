import copy
import os
from datetime import datetime
from typing import Union, Optional, Dict
from itertools import product

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnvTimestep
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from zoo.classic_control.cartpole.envs.cartpole_lightzero_env import CartPoleEnv

@ENV_REGISTRY.register('pendulum_lightzero')
class PendulumEnv(CartPoleEnv):
    """
    LightZero version of the classic Pendulum environment. This class includes methods for resetting, closing, and
    stepping through the environment, as well as seeding for reproducibility, saving replay videos, and generating random
    actions. It also includes properties for accessing the observation space, action space, and reward space of the
    environment.
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Get the default configuration of the BipedalWalker environment.
        Returns:
            - cfg (:obj:`EasyDict`): Default configuration dictionary.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        # (str) The gym environment name.
        env_name="Curling",
        replay_path=None,
        continuous=True,
    )

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialize the BipedalWalker environment with the given config dictionary.
        Arguments:
            - cfg (:obj:`dict`): Configuration dictionary.
        """
        self._cfg = cfg
        self._init_flag = False
        self._continuous = True
        self.prob_random_agent = cfg.prob_random_agent
        self._observation_space = gym.spaces.Box(low=0,high=1,shape=(5,96,96),dtype=np.float32)
        self._action_space = gym.spaces.Box(low=-1,high=1,shape=(3,),dtype=np.float32)
        self._action_space.seed(0)  # default seed
        self._reward_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.players = [1, 2]
        self._current_player = 0
        self.battle_mode = cfg.battle_mode

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment. If it hasn't been initialized yet, this method also handles that. It also handles seeding
        if necessary. Returns the first observation.
        """
        if not self._init_flag:
            try:
                self._env = gym.make('Pendulum-v1', render_mode="rgb_array")
            except:
                self._env = gym.make('Pendulum-v0', render_mode="rgb_array")
            if self._replay_path is not None:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                video_name = f'{self._env.spec.id}-video-{timestamp}'
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix=video_name
                )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._seed = self._seed + np_seed
            self._action_space.seed(self._seed)
            obs, _ = self._env.reset(seed=self._seed)
        elif hasattr(self, '_seed'): 
            self._action_space.seed(self._seed)
            obs, _ = self._env.reset(seed=self._seed)
        else:
            obs, _ = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        self._eval_episode_return = 0.

        if not self._continuous:
            action_mask = np.ones(self.discrete_action_num, 'int8')
        else:
            action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return obs

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        """
        Overview:
            Step the environment forward with the provided action. This method returns the next state of the environment
            (observation, reward, done flag, and info dictionary) encapsulated in a BaseEnvTimestep object.
        Arguments:
            - action (:obj:`Union[int, np.ndarray]`): The action to be performed in the environment.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): An object containing the new observation, reward, done flag,
              and info dictionary.

        .. note::
            - If the environment requires discrete actions, they are converted to float actions in the range [-1, 1].
            - If action scaling is enabled, continuous actions are scaled into the range [-2, 2].
            - For each step, the cumulative reward (`_eval_episode_return`) is updated.
            - If the episode ends (done is True), the total reward for the episode is stored in the info dictionary
              under the key 'eval_episode_return'.
            - If the environment requires discrete actions, an action mask is created, otherwise, it's None.
            - Observations are returned in a dictionary format containing 'observation', 'action_mask', and 'to_play'.
        """
        if isinstance(action, int):
            action = np.array(action)
        # if require discrete env, convert actions to [-1 ~ 1] float actions
        if not self._continuous:
            action = (action / (self.discrete_action_num - 1)) * 2 - 1
        # scale the continous action into [-2, 2]
        if self._act_scale:
            action = affine_transform(action, min_val=self._env.action_space.low, max_val=self._env.action_space.high)
        obs, rew, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        self._eval_episode_return += rew
        obs = to_ndarray(obs).astype(np.float32)
        # wrapped to be transferred to an array with shape (1,)
        rew = to_ndarray([rew]).astype(np.float32)

        if done:
            info['eval_episode_return'] = self._eval_episode_return

        if not self._continuous:
            action_mask = np.ones(self.discrete_action_num, 'int8')
        else:
            action_mask = None
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return BaseEnvTimestep(obs, rew, done, info)

    def random_action(self) -> np.ndarray:
        """
         Generate a random action using the action space's sample method. Returns a numpy array containing the action.
         """
        if self._continuous:
            random_action = self.action_space.sample().astype(np.float32)
        else:
            random_action = self.action_space.sample()
            random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    def __repr__(self) -> str:
        """
        String representation of the environment.
        """
        return "LightZero Pendulum Env({})".format(self._cfg.env_id)