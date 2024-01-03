import sys
from pathlib import Path

import numpy as np
from scipy.spatial import distance

from dc3client import SocketClient

sys.path.append("./src")
from zoo.curling.src.util import calc_distance_and_neighbor_point


class Brain:
    def __init__(self, cli: SocketClient) -> None:
        self.myteam = cli.get_my_team()
        self.x_min = -4.75 / 2
        self.x_max = 4.75 / 2
        self.y_max = 40.234
        self.y_min = 0
        self.radius = 1.829
        self.center_x = 0
        self.center_y = 38.405
        self.AB_mat_dict = self.load_npz()
        self.stone_size = 0.278  # 適当

    def calc_distance(self, x, y):
        return np.sqrt((x - self.center_x) ** 2 + (y - self.center_y) ** 2)
    
    def calc_distance_arr(self, xy):
        """
        centerとの距離を計算
        xy : numpy array shape (n,2)
        """

        center = np.array([[self.center_x, self.center_y]])
        dist = distance.cdist(center, xy, metric="euclidean")[0]

        return dist

    def get_stone_position_info(self, state):
        """
        stateから石の場所を返す, noneは削除される
        """
        _x = [_stone.position[0].x for _stone in state.stones.team0]
        _y = [_stone.position[0].y for _stone in state.stones.team0]
        # noneを削除
        stones_team0 = np.array([_x, _y], dtype=np.float32).T
        stones_team0 = stones_team0[~np.isnan(stones_team0[:, 0])]

        _x = [_stone.position[0].x for _stone in state.stones.team1]
        _y = [_stone.position[0].y for _stone in state.stones.team1]
        stones_team1 = np.array([_x, _y], dtype=np.float32).T
        stones_team1 = stones_team1[~np.isnan(stones_team1[:, 0])]

        if self.myteam == "team0":
            my_stones, your_stones = stones_team0, stones_team1
        else:
            my_stones, your_stones = stones_team1, stones_team0
        return my_stones, your_stones
    
    def load_npz(self, npz_dir=Path("./model/position")):
        """
        位置と速度の対応行列 Y(位置) = A X(速度) + B
        """
        npz_list = ["AB_fin.npz", "AB_vel0.npz", "AB_vel1.npz", "AB_vel2.npz"]
        mode_list = ["mode0", "mode1", "mode2", "mode3"]
        AB_mat_dict = {}

        for _npz, _mode in zip(npz_list, mode_list):
            npz_path = npz_dir / _npz
            AB_npz = np.load(npz_path)
            A_ccw = AB_npz["A_ccw"]
            B_ccw = AB_npz["B_ccw"]
            A_cw = AB_npz["A_cw"]
            B_cw = AB_npz["B_cw"]

            AB_mat_dict[_mode] = [A_ccw, B_ccw, A_cw, B_cw]

        return AB_mat_dict
    
    def pred_input(self, pos_x, pos_y, mode="mode0"):
        """
        位置とモードから入力を予測する

        mode
        0:とめる
        1:弱くぶつける(velocity 0.05)
        2:ぶつける(velocity 0.1)
        3:強くぶつける(velocity 0.2)
        """
        A_ccw, B_ccw, A_cw, B_cw = self.AB_mat_dict[mode]
        if pos_x < 0:
            A, B = A_ccw, B_ccw
            r = "ccw"
        else:
            A, B = A_cw, B_cw
            r = "cw"
        _Y = np.array([[pos_x], [pos_y]])
        _A_inv = np.linalg.pinv(A)
        _X_hat = np.dot(_A_inv, (_Y - B))

        return _X_hat[0][0], _X_hat[1][0], r
    
    def pred_position(self, v_x, v_y, rotation, mode="mode0"):
        """
        入力とモードから位置を予測する

        mode
        0:とめる
        1:弱くぶつける(velocity 0.05)
        2:ぶつける(velocity 0.1)
        3:つよくぶつける(velocity 0.2)
        """

        A_ccw, B_ccw, A_cw, B_cw = self.AB_mat_dict[mode]
        if rotation == "ccw":
            A, B = A_ccw, B_ccw
        else:
            A, B = A_cw, B_cw
        _X = np.array([[v_x], [v_y]])
        _Y_hat = np.dot(A, _X) + B

        return _Y_hat[0][0], _Y_hat[1][0]
    
    def pred_position_allmode(self, v_x, v_y, rotation):
        """
        すべてのモードでの位置予測結果を返す
        """
        x0, y0 = self.pred_position(v_x, v_y, rotation, mode="mode0")
        x1, y1 = self.pred_position(v_x, v_y, rotation, mode="mode1")
        x2, y2 = self.pred_position(v_x, v_y, rotation, mode="mode2")
        x3, y3 = self.pred_position(v_x, v_y, rotation, mode="mode3")
        return [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
    
    def calc_distance_from_orbit(self, v_x, v_y, rotation, pos_x, pos_y):
        """
        衝突するかどうかを判定するための部分関数
        位置と入力が与えられたときに、予測軌跡と位置を最短距離で返す
        """
        xy_list = self.pred_position_allmode(v_x, v_y, rotation)
        min_distance = 1000
        for xy0, xy1 in zip(xy_list[:-1], xy_list[1:]):
            xy0, xy1 = np.array(xy0), np.array(xy1)
            _pos = np.array((pos_x, pos_y))
            _, distance = calc_distance_and_neighbor_point(xy0, xy1, _pos)
            if min_distance > distance:
                min_distance = distance
        return min_distance
    
    def will_collisition(self, v_x, v_y, rotation, my_stones_pos, your_stones_pos):
        """
        衝突するかどうかを判定する
        軌跡との各石の位置との最小距離が石のサイズよりも小さかったらTrue, 大きかったらFalse
        """
        if (len(my_stones_pos) == 0) and (len(your_stones_pos) == 0):
            return False
        elif len(my_stones_pos) == 0:
            stones_pos = your_stones_pos
        elif len(your_stones_pos) == 0:
            stones_pos = my_stones_pos
        else:
            stones_pos = np.concatenate([my_stones_pos, your_stones_pos])

        will_collistion_flag = False

        for _pos in stones_pos:
            distance = self.calc_distance_from_orbit(
                v_x, v_y, rotation, _pos[0], _pos[1]
            )

            if distance < self.stone_size:
                will_collistion_flag = True
        
        return will_collistion_flag