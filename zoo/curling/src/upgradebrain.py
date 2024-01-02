import random
import sys
from typing import Literal

import numpy as np
from numpy import typing as npt

from zoo.curling.src.brain import Brain
from dc3client import SocketClient
from dc3client.models import State
from zoo.curling.src.util import calc_distance_and_neighbor_point

sys.path.append("./src")


class NewBrain(Brain):
    def __init__(self, cli: SocketClient):
        super().__init__(cli)
        self.radius = 1.829

    def visualize(self, state:State, my_stones_pos:npt.NDArray[np.float64], your_stones_pos:npt.NDArray[np.float64]):
        print("-------------")
        _end, _shot = state.end, state.shot
        print(f"#end :{_end} #team:{self.myteam} #shot : {_shot}")
        print("my stone")
        print(my_stones_pos)
        print("your stone")
        print(your_stones_pos)

    def decide(self, cli:SocketClient, mode="mode2"):
        state = cli.match_data.update_list[-1].state
        my_stones_pos, your_stones_pos = self.get_stone_position_info(state)
        self.visualize(state, my_stones_pos, your_stones_pos)
        if state.shot == 0:
            # 初めのショットはガードゾーンへ置く
            x, y, r = (
                self.center_x,
                self.center_y - self.stone_size - self.radius,
                "ccw",
            )
            return self.pred_input(x, y, r, mode="mode0")
        if len(your_stones_pos) == 0:
            return self.decide_non_enemy_stone(state, my_stones_pos)
        is_first = state.shot % 2
        distance = self.calc_distance_arr(your_stones_pos)
