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
        if win_team == your_team:
            pass
        else:
            value *= -1
        return value*5

        

