import numpy as np
from numpy.linalg import norm

def calc_distance_and_neighbor_point(a, b, p):
    ap = p - a
    ab = b - a
    ba = a - b
    bp = p - b
    if np.dot(ap, ab) < 0: