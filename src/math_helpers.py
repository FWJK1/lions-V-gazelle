import constants as c
import numpy as np


## math helpers
def vector_between(pos_a, pos_b):
    return (pos_b - pos_a + c.WORLD / 2) % c.WORLD - c.WORLD / 2


def angle_diff(angle1, angle2):
    diff = angle1 - angle2
    return (diff + np.pi) % (2 * np.pi) - np.pi


def random_unit_vector() -> np.ndarray:
    v = np.random.standard_normal(2)
    return v / np.linalg.norm(v)


def rotate_90(vec):
    x, y = vec
    return np.array([y, -x])


def rand_scale(vec):
    return vec * np.random.random()


def invert(vec: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(vec)
    if mag == 0:
        return np.zeros(2)
    return (vec / mag) * (c.MAX_NORM - mag)
