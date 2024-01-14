import gymnasium as gym
import pickle
import yaml
import copy
from zoo.curling.src.dcsimulate import Simulate
from zoo.curling.src.rvalue import Rvalue_fast, Rvalue_team0, Rvalue_end
from zoo.curling.src.data_df import stones_fast
import numpy as np
from zoo.curling.src.map_array import Stone_map_gaussian

class MyEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, team: str = "team0") -> None:
        super(MyEnv, self).__init__()
        self.stone2map = Stone_map_gaussian((96, 96))
        self.final_end = 10
        self.shot_end = 0

        with open("/dc3/config/config.yml", 'r') as yml:
            cfg = yaml.safe_load(yml)
        
        with open("/dc3/gamedata/init_game_state.pkl", "rb") as f:
            self.init_game_state = pickle.load(f)
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,96,96), dtype=np.float32)

        if team == "team0":
            port = 10000
        else:
            port = 10001
        self.initial_map = self.stone2map(None)
        self.game_state = copy.deepcopy(self.init_game_state)
        self.port = port
        self.simulate = Simulate(port=port)
        #self.reward = Rvalue_team0()
        self.reward = Rvalue_end()
    
    def enemy_turn_get_obs(self):
        state = self.simulate.get_state()
        return self.state_to_obs(state)
    
    def reset(self):
        self.game_state = self.init_game_state
        # hammerじゃない方が先行になるはず
        my_team = self.game_state["hammer"] != "team1"
        return self.initial_map.copy(), my_team
    
    def reset_state(self, state):
        self.game_state = state
    
    def state_to_obs(self, next_state):
        stone_array = stones_fast(next_state)
        if next_state["shot"] == 0 and next_state["end"] > 0:
            reward = self.reward(next_state, self.game_state)
        else:
            if next_state["stones"] == self.game_state["stones"]:  # 盤面に変化が無いとき
                reward = -1
            else:
                reward = 0
        # チャンネルの意味: 後攻の石|先攻の石|ハウスのガウス|ショット|手番|
        maps = self.stone2map(stone_array, next_state["hammer"] == "team1", next_state["shot"])

        # 本来の使い方ではないが、ゲームが終了したか確認する。
        done = False
        if next_state["end"] == self.final_end:
            done = True
        
        self.game_state = next_state  # game_state を更新
        info = {}
        info["shot"] = next_state["shot"]
        info["hammer"] = next_state["hammer"]
        return maps, reward, done, False, info
    
    def step(self, action, is_shot = False):
        x = action[0]*0.3
        y = action[1]*0.8 + 3
        r = "ccw" if action[2] > 0 else "cw"
        now_hammer = self.game_state["hammer"] == "team1"

        next_state = self.simulate(x, y, r, self.game_state, is_shot)

        return self.state_to_obs(next_state)
    



