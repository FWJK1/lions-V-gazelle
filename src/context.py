from dataclasses import dataclass, field
import numpy as np
from config_parser import Config
from itertools import combinations
import math_helpers as m


@dataclass
class Context:
    config: Config
    last: np.ndarray
    gazelle: np.ndarray
    vecs_to_gazelle: np.ndarray
    dist_to_gazelle: np.ndarray
    nearest: np.ndarray
    caught_gazelle: bool
    lion: np.ndarray
    rlion: np.ndarray
    llion: np.ndarray
    lion1: np.ndarray
    lion2: np.ndarray
    lion3: np.ndarray
    lion4: np.ndarray
    heading: np.ndarray = field(default_factory=lambda: np.array([0, 0]))


def get_gazelle_vectors(positions: np.ndarray, c: Config) -> tuple:
    gazelle = positions[0]
    vecs_to_gazelle = np.array(
        [m.vector_between(lion, gazelle, c) for lion in positions[1:]]
    )
    dist_to_gazelle = np.linalg.norm(vecs_to_gazelle, axis=1)
    nearest = np.argmin(dist_to_gazelle)
    caught_gazelle = True if dist_to_gazelle[nearest] < 1 else False
    nearest = vecs_to_gazelle[nearest]
    return vecs_to_gazelle, dist_to_gazelle, nearest, caught_gazelle


def update_gazelle_ctx(
    positions: np.ndarray, previous_ctxs: list[Context], c: Config
) -> list[Context]:
    """generate spatial contexts for the gazelle, which is shared.

    Args:
        positions (np.ndarray): _description_
        previous_ctxs (list[Context]): _description_

    Returns:
        list[Context]: _description_
    """
    vecs_to_gazelle, dist_to_gazelle, nearest, caught_gazelle = get_gazelle_vectors(
        positions, c
    )
    ctxs = []
    ctxs.append(None)
    for lion_idx in range(1, 5):
        ctx = Context(
            config=c,
            last=previous_ctxs[lion_idx].last,
            gazelle=vecs_to_gazelle[lion_idx - 1],
            vecs_to_gazelle=vecs_to_gazelle,
            dist_to_gazelle=dist_to_gazelle,
            nearest=nearest,
            caught_gazelle=caught_gazelle,
            lion=previous_ctxs[lion_idx].lion,
            rlion=previous_ctxs[lion_idx].rlion,
            llion=previous_ctxs[lion_idx].llion,
            lion1=previous_ctxs[lion_idx].lion1,
            lion2=previous_ctxs[lion_idx].lion2,
            lion3=previous_ctxs[lion_idx].lion3,
            lion4=previous_ctxs[lion_idx].lion4,
        )
        ctxs.append(ctx)
    return ctxs


def update_ctxs(
    positions: np.ndarray, previous_ctxs: list[Context] | None, c: Config
) -> list[Context]:
    """generate spatial contexts for each lion and the gazelle

    Args:
        positions np.ndarray: shape (5, 2), gazelle=0, then each lion indexed by number
        ctxs np.ndarray: shape (5, 2), gazelle (dummy)=0, then each lion indexed by number


    Returns:
        list[dict]: list of contexts, indexed by lion number (0 is a dummy for gazelle)
    """

    vecs_to_gazelle, dist_to_gazelle, nearest, caught_gazelle = get_gazelle_vectors(
        positions, c
    )

    # set defaults to zero
    norms = np.zeros((5, 5))
    angles = np.zeros((5, 5))
    vecs = np.zeros((5, 5, 2))
    for a, b in combinations([1, 2, 3, 4], 2):
        vec = m.vector_between(positions[a], positions[b], c)
        norm = np.linalg.norm(vec)
        vecs[a, b], vecs[b, a] = vec, -vec
        norms[a, b] = norms[b, a] = norm
        angles[a, b] = np.arctan2(vec[1], vec[0])
        angles[b, a] = np.arctan2(-vec[1], -vec[0])

    ctxs = []
    ctxs.append(None)

    for lion_idx in range(1, 5):
        last = (
            previous_ctxs[lion_idx].heading
            if previous_ctxs
            else np.random.random_sample(2) * np.array([c.width, c.height])
        )
        other_idxs = [other for other in range(1, 5) if other != lion_idx]
        closest_lion = vecs[
            lion_idx, other_idxs[np.argmin(norms[lion_idx, other_idxs])]
        ]

        heading_angle = np.arctan2(last[1], last[0])
        rel_angles = np.array(
            [
                m.angle_diff(angles[lion_idx, other], heading_angle)
                for other in other_idxs
            ]
        )
        rlion = vecs[lion_idx, other_idxs[np.argmin(rel_angles)]]
        llion = vecs[lion_idx, other_idxs[np.argmax(rel_angles)]]

        ctx = Context(
            config=c,
            last=last,
            gazelle=vecs_to_gazelle[lion_idx - 1],
            vecs_to_gazelle=vecs_to_gazelle,
            dist_to_gazelle=dist_to_gazelle,
            nearest=nearest,
            caught_gazelle=caught_gazelle,
            lion=closest_lion,
            rlion=rlion,
            llion=llion,
            lion1=vecs[lion_idx, 1],
            lion2=vecs[lion_idx, 2],
            lion3=vecs[lion_idx, 3],
            lion4=vecs[lion_idx, 4],
        )
        ctxs.append(ctx)

    return ctxs
