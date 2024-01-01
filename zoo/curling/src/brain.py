import sys
from pathlib import Path

import numpy as np
from scipy.spatial import distance

from dc3client import SocketClient

sys.path.append("./src")
from zoo.curling.src.util import calc_distance_and_neighbor_point
