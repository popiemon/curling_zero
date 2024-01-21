#! Python3

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple
from scipy.signal import windows
import cv2

class Stone_map_gaussian:
    def __init__(self, map_size = (128, 224)):
        self.stone_size = 290
        self.map_size = map_size
        len_circle = 1829 + self.stone_size
        self.x_size = 4750
        self.y_size = 8230 + self.stone_size
        center_y = 38405-32004
        center_x = self.x_size // 2
        center_array = np.zeros([self.y_size, self.x_size], dtype=np.float32)
        gk = self.gaussian_kernel(len_circle*2+1, 1219)
        gk[gk < gk[0, len_circle]] = 0
        center_array[
            center_y - len_circle - 1 : center_y + len_circle, center_x - len_circle : center_x + len_circle + 1,
        ] = gk
        self.center_array = cv2.resize(center_array, map_size)
        self.stone_gk = self.gaussian_kernel(self.stone_size+1, 50)
        self.stone_gk[self.stone_gk < self.stone_gk[0, self.stone_size//2]] = 0
    
    def gaussian_kernel(self, n, std):
        gaussian1D = windows.gaussian(n, std)
        gaussian2D = np.outer(gaussian1D, gaussian1D)
        gaussian2D /= gaussian2D.max()
        return gaussian2D
    
    def update_map(self, maps):
        sheat = np.zeros([self.y_size, self.x_size], dtype=np.float32)
        half_size = self.stone_size//2
        for array in maps:
            x = ((array[1] + 2.375) // 0.001).astype(np.int16)
            y = ((array[2] - 32.004) // 0.001).astype(np.int16)
            try:
                sheat[
                    max(y - half_size, 0) : min(y + half_size + 1, self.y_size),
                    max(x - half_size, 0) : min(x + half_size + 1, self.x_size),
                ] = self.stone_gk[
                    max(half_size - y, 0) : min(self.y_size - y + half_size, self.stone_size+1),
                    max(half_size - x, 0) : min(self.x_size - x + half_size, self.stone_size+1),
                ]
            except:
                continue
        return cv2.resize(sheat, self.map_size)
    
    def __call__(self, in_map: npt.NDArray[np.float32]|None, hammer = 0, shot = 0) -> np.array:
        """2D array を作成。

        Args:
            in_map (npt.NDArray[np.float32] | None): _description_
            hammer (int, optional): _description_. Defaults to 0.
            shot (int, optional): _description_. Defaults to 0.

        Returns:
            np.array: stonesの位置情報
        """
        map = np.zeros((self.map_size[1], self.map_size[0], 5), dtype=np.float32)
        my_team = (hammer) ^ ((shot+1) % 2)
        map[:,:,2] = self.center_array
        map[:,:,4] = my_team

        if in_map is None:
            return map
        stonemap0 = in_map[in_map[:,0] == hammer]
        stonemap1 = in_map[in_map[:,0] != hammer]

        if len(stonemap0)>0:
            map[:,:,0] = self.update_map(stonemap0)
        if len(stonemap1)>0:
            map[:,:,1] = self.update_map(stonemap1)
        map[:,:,3] = shot/15

        return map
    
    def mch_map(self, in_map: npt.NDArray[np.float32]|None, hammer = 0, shot = 0) -> np.array:
        """__call__関数がorg。アレンジしたver

        Args:
            in_map (npt.NDArray[np.float32] | None): _description_
            hammer (int, optional): _description_. Defaults to 0.
            shot (int, optional): _description_. Defaults to 0.

        Returns:
            np.array: _description_
        """
        map = np.zeros((self.map_size[1], self.map_size[0], 5), dtype=np.float32)
        my_team = (hammer) ^ ((shot+1) % 2)  # 0 or 1
        map[:,:,2] = self.center_array
        map[:,:,4] = my_team  # 0 or 1

        if in_map is None:
            return map
        mystonemap = in_map[in_map[:,0] == my_team] 
        enstonemap = in_map[in_map[:,0] != my_team]

        if len(mystonemap)>0:
            map[:,:,0] = self.update_map(mystonemap)
        if len(enstonemap)>0:
            map[:,:,1] = self.update_map(enstonemap)
        map[:,:,3] = shot/15

        return map
        