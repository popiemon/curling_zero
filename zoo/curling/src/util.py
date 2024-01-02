import numpy as np
from numpy.linalg import norm

def calc_distance_and_neighbor_point(a, b, p):
    ap = p - a
    ab = b - a
    ba = a - b
    bp = p - b
    if np.dot(ap, ab) < 0:
        distance = norm(ap)
        neighbor_point = a
    elif np.dot(bp, ba) < 0:
        distance = norm(p -b)
        neighbor_point = b
    else:
        ai_norm = np.dot(ap, ab) / norm(ab)
        neighbor_point = a+ (ab) / norm(ab) * ai_norm
        distance = norm(p - neighbor_point)
    return (neighbor_point, distance)
