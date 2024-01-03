import copy
import os
from datetime import datetime
from typing import Union, Optional, Dict
from itertools import product

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnvTimestep, BaseEnv
from ding.envs.common import affine_transform
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from zoo.curling.src.alphaenv import MyEnv

@ENV_REGISTRY.register('Curling')
class CurlingcEnv(BaseEnv):
    """
    Overview:
        The modified BipedalWalker environment with manually discretized action space. For each dimension, equally dividing the
        original continuous action into ``each_dim_disc_size`` bins and using their Cartesian product to obtain
        handcrafted discrete actions.
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
            self._env = MyEnv()
            self._init_flag = True
        obs, self.start_player_index = self._env.reset()
        self._observation_space = self._env._observation_space
        self._eval_episode_return = 0
        obs = to_ndarray(obs)
        self._current_player = self.players[self.start_player_index]
        if self.battle_mode == "self_play_mode":
            obs = {'observation': obs, 'action_mask': None, "current_player_index": self.start_player_index, 'to_play': self.current_player}
        else:
            obs = {'observation': obs, 'action_mask': None, "current_player_index": self.start_player_index, 'to_play': -1} 

        return obs

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        """
        Overview:
            Perform a step in the environment using the provided action, and return the next state of the environment.
            The next state is encapsulated in a BaseEnvTimestep object, which includes the new observation, reward,
            done flag, and info dictionary.
        Arguments:
            - action (:obj:`Union[int, np.ndarray]`): The action to be performed in the environment. If the action is
              a 1-dimensional numpy array, it is squeezed to a 0-dimension array.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): An object containing the new observation, reward, done flag,
              and info dictionary.
        .. note::
            - The cumulative reward (`_eval_episode_return`) is updated with the reward obtained in this step.
            - If the episode ends (done is True), the total reward for the episode is stored in the info dictionary
              under the key 'eval_episode_return'.
            - An action mask is created with ones, which represents the availability of each action in the action space.
            - Observations are returned in a dictionary format containing 'observation', 'action_mask', and 'to_play'.
        """
        if self.battle_mode == "self_play_mode":
            if np.random.rand() < self.prob_random_agent:
                action = self.random_action()
            
            if isinstance(action, np.ndarray) and action.shape == (1,):
                action = action.squeeze()  # 0-dim array
            
            obs, rew, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated

            self.current_player = self.players[(info["hammer"] != "team1") ^ (info["shot"] % 2)]
            if rew != 0:
                self._eval_episode_return += rew
            if done:
                info['eval_episode_return'] = self._eval_episode_return
                # print(info['eval_episode_return'])
        
            action_mask = None
            obs = {'observation': obs, 'action_mask': action_mask, 'current_player_index': self.players.index(self.current_player), 'to_play': self.current_player}
        else:
            # LightZero Turn
            if isinstance(action, np.ndarray) and action.shape == (1,):
                action = action.squeeze()  # 0-dim array
            obs, rew, terminated, truncated, info = self._env.step(action, True)
            done = truncated
            self.current_player = self.to_play
            if rew != 0:
                self._eval_episode_return += rew
            if terminated:
                info['eval_episode_return'] = self._eval_episode_return
                obs = {'observation': obs, 'action_mask': None, 'to_play': -1}
                return BaseEnvTimestep(obs, rew, done, info)
            
            # bot Turn
            # bot_action = self.house_shot()

            obs, rew, terminated, truncated, info = self._env.enemy_turn_get_obs()
            if rew != 0:
                self._eval_episode_return += rew
            if terminated:
                info['eval_episode_return'] = self._eval_episode_return
            obs = {'observation': obs, 'action_mask': None, 'to_play': -1}
        
        return BaseEnvTimestep(obs, rew, done, info)

    def close(self) -> None:
        """
        Close the environment, and set the initialization flag to False.
        """
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the seed for the environment's random number generator. Can handle both static and dynamic seeding.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        """
        Enable the saving of replay videos. If no replay path is given, a default is used.
        """
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        """
         Generate a random action using the action space's sample method. Returns a numpy array containing the action.
         """
        if np.random.rand() < 0.7:
            # ハウスに入れるショット、とりあえず高確率がええやろってことで
            random_action = self.house_shot()
        else:
            # ランダムなショット
            random_action = self.action_space.sample()
            # random_action = to_ndarray(random_action, dtype=np.int64)
        return random_action
    
    def house_shot(self) -> np.ndarray:
        x = np.random.rand()*0.6 + 0.1
        y = -0.7-np.random.rand()*0.1
        r = np.random.rand()-0.5
        if r < 0:
            x = -x
        return np.array([x,y,r])

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Property to access the observation space of the environment.
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Property to access the action space of the environment.
        """
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        """
        Property to access the reward space of the environment.
        """
        return self._reward_space

    def __repr__(self) -> str:
        """
        String representation of the environment.
        """
        return "LightZero Curling Env"
    
    @property
    def to_play(self):
        return self.players[0] if self.current_player == self.players[1] else self.players[1]
    
    @property
    def current_player(self):
        return self._current_player
    
    @current_player.setter
    def current_player(self, value):
        self._current_player = value