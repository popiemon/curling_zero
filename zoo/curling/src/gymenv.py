import gymnasium as gym
import pickle
import yaml

from src.dcsimulate import Simulate
from src.rvalue import Rvalue
import src.data_df as data_df

class MyEnv(gym.Env):
    def __init__(self, team: str = "team0"):
        super(MyEnv, self).__init__()

        self.final_end = 10

        with open("config/config.yml", "r") as yml:
            cfg = yaml.safe_load(yml)
        
        with open("gamedata/init_game_state.pickle", "rb") as f:
            self.init_game_state = pickle.load(f)
        
        if team == "team0":
            port = 10000
        else:
            port = 10001

        self.game_state = self.init_game_state

        self.simulate = Simulate(port = port)
        self.reward = Rvalue(team=team, point=cfg["reward"]["inside_stone"])

    def reset(self):
        self.game_state = self.init_game_state
        return self.game_state
    
    def step(self, action):
        x = action[0]
        y = action[1]
        r = action[2]

        next_state = self.simulate(x, y, r, self.game_state)

        stone_df = data_df.stones(next_state)
        reward = self.reward(stone_df, omega=0.5)

        done = False
        if next_state["end"] > self.final_end:
            done = True

        self.game_state = next_state # game_state を更新

        return next_state, reward, done
