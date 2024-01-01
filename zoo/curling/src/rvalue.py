#! Python3

from typing import Any
import numpy as np
import pandas as pd

class Rvalue_team0:
    def __call__(self, state) -> float:
        """チーム○から見た得点を返す
        """
        # 1end進んでいるためhammerじゃない方がハンマー
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

