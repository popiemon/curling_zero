#! Python3

from typing import Any
import numpy as np
import numpy.typing as npt
import pandas as pd

class Rvalue_team0:
    def __call__(self, state) -> float:
        """チーム○から見た得点を返す
        """
        end = state["end"] - 1
        hammer_point = state["scores"]["team0"][end]
        other_point = state["scores"]["team1"][end]
        return hammer_point - other_point
    
class Rvalue_fast:
    def __call__(self, state, hammer) -> float:
        """結果からリワードを返す。
        """
        key = list(state["scores"].keys())
        # 1end進んでいるためhammerじゃない方がハンマー
        end = state["end"] - 1
        hammer_point = state["scores"][key[hammer]][end]
        other_point = state["scores"][key[not hammer]][end]
        return hammer_point - other_point - 0.9

class Rvalue_end:
    def __call__(self, next_state, state) -> float:
        """1end勝った方に+の, 負けた方に-のリワードを返す

        Args:
            state (_type_): game_state

        Returns:
            float: reward
        """
        end = state["end"]
        your_team = state["hammer"]  # "team0" or "team1"
        # 1end進んでいるため、hammerじゃない方が、1end前の勝者
        lose_team = next_state["hammer"]  # "team0" or "team1"
        if lose_team is None:  # game終了
            win_team = next_state["game_result"]["winner"]
            if win_team == "team0":
                lose_team = "team1"
            elif win_team == "team1":
                lose_team = "team0"
            win_point = sum(next_state["scores"][win_team])
            lose_point = sum(next_state["scores"][lose_team])
        else:  # game中
            if lose_team == "team0":
                win_team = "team1"
            elif lose_team == "team1":
                win_team = "team0"
        
            win_point = next_state["scores"][win_team][end]
            lose_point = next_state["scores"][lose_team][end]
        
        if win_point is None:
            win_point = 0
        if lose_point is None:
            lose_point = 0
        value = win_point - lose_point
        if win_team == your_team:  # 勝った時
            pass
        else:  # 負けた時
            value = 0
        return value*5

class Rvalue_stones:
    def __init__(self):
        self.center_x = 0  # house center
        self.center_y = 38.405  # house center
        self.max_reward = 10
    
    def __call__(self, next_state, state):
        # my team を算出
        if state["shot"] % 2 != 0:  # 奇数
            my_team = state["hammer"]
            if my_team == "team0":
                en_team = "team1"
            else:
                en_team = "team0"
        else:  # 偶数
            if state["hammer"] == "team0":
                my_team = "team1"
                en_team = "team0"
            else:
                my_team = "team0"
                en_team = "team1"
        my_stones = state["stones"][my_team]
        en_stones = state["stones"][en_team]

        my_next_stones = next_state["stones"][my_team]
        en_next_stones = next_state["stones"][en_team]

        for stone in my_stones:
            dists = []
            if stone is not None:
                stone_x = stone["position"]["x"]
                stone_y = stone["position"]["y"]
                dist = np.sqrt((self.center_x - stone_x)**2 + (self.center_y - stone_y)**2)
                dists.append(dist)

        for stone in my_next_stones:
            next_dists = []
            if stone is not None:
                stone_x = stone["position"]["x"]
                stone_y = stone["position"]["y"]
                dist = np.sqrt((self.center_x - stone_x)**2 + (self.center_y - stone_y)**2)
                next_dists.append(dist)
        
        if (len(dists) == 0) and (len(next_dists) == 0):
            reward = 0
        elif (len(dists) == 0) and (len(next_dists) != 0):
            reward = self.max_reward - min(next_dists)
        elif (len(dists) != 0) and (len(next_dists) == 0):
            reward = 0
        else:
            reward = self.max_reward - min(next_dists)

        return reward 