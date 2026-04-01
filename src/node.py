"""
**Author**:
    Fitz Koch
**Created**:
    2026-04-01
**Description**:
    Code written by Fitz Koch with some bug checking from Claude
"""

import random
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from datetime import datetime
from typing import Callable

import numpy as np

import constants as c
from viz import viz_sim


OPERATORS: dict[str, tuple[int, Callable]] = {
    # --- Terminals (arity 0): look up pre-computed value in ctx ---
    # standard
    "last": (0, lambda ctx: ctx.last),
    "rand-dir": (0, lambda ctx: random_unit_vector()),
    "gazelle": (0, lambda ctx: ctx.gazelle),
    # deictic
    "nearest": (0, lambda ctx: ctx.nearest),  # nearest-to-gazelle lion -> gazelle
    "lion": (0, lambda ctx: ctx.lion),  # vector to nearest neighbor
    "rlion": (0, lambda ctx: ctx.rlion),  # clockwise sweep neighbor
    "llion": (0, lambda ctx: ctx.llion),  # counterclockwise sweep neighbor
    # name-based (self excluded at ctx build time)
    "lion-1": (0, lambda ctx: ctx.lion1),
    "lion-2": (0, lambda ctx: ctx.lion2),
    "lion-3": (0, lambda ctx: ctx.lion3),
    "lion-4": (0, lambda ctx: ctx.lion4),
    # --- Internal nodes ---
    "+": (2, lambda ctx, a, b: a + b),
    "-": (1, lambda ctx, a: -a),
    "*2": (1, lambda ctx, a: a * 2),
    "/2": (1, lambda ctx, a: a / 2),
    "->90": (1, lambda ctx, a: rotate_90(a)),
    "rand": (1, lambda ctx, a: rand_scale(a)),
    "inv": (1, lambda ctx, a: invert(a)),
    "ifdot": (4, lambda ctx, a, b, c, d: c if np.dot(a, b) >= 0 else d),
    "if>=": (
        4,
        lambda ctx, a, b, c, d: c if np.linalg.norm(a) >= np.linalg.norm(b) else d,
    ),
}
# fmt: on

# Terminal sets per sensing mode
BASE_TERMINALS = ["last", "rand-dir", "gazelle"]
DEICTIC_TERMINALS = BASE_TERMINALS + ["nearest", "lion", "rlion", "llion"]
NAME_TERMINALS = BASE_TERMINALS + ["lion-1", "lion-2", "lion-3", "lion-4"]

INTERNALS = ["+", "-", "*2", "/2", "->90", "rand", "inv", "ifdot", "if>="]


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


class Node:
    def __init__(self, operation: str, children: list["Node"] | None = None) -> None:
        self.operation = operation
        self.children = children or []

    def process_inputs(self, ctx: Context) -> np.ndarray:
        arity, func = OPERATORS[self.operation]
        child_vals = [child.process_inputs(ctx) for child in self.children]

        return func(ctx, *child_vals)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def __repr__(self) -> str:
        if not self.children:
            return f"{self.operation}"
        child_vals = " ".join([c.operation for c in self.children])
        return f"{self.operation} on {[c for c in child_vals]}"


def random_lion(terminals, max_depth=c.MAX_DEPTH, depth=0) -> Node:
    """build a random tree of node units; recursive, but and when called from root returns a Lion, basically

    Args:
        terminals (_type_): the terminals for this type of lion
        max_depth (_type_, optional): how deep we can go. Defaults to c.MAX_DEPTH.
        depth (int, optional): current depth. Defaults to 0.

    Returns:
        Node: the root of a lion tree
    """
    force_terminal = depth >= max_depth
    pick_terminal = force_terminal or (
        depth > 0 and np.random.random_sample() < c.EARLY_TERMINAL_P
    )

    if pick_terminal:
        return Node(random.choice(terminals))

    operation = random.choice(INTERNALS)
    arity, _ = OPERATORS[operation]
    children = [random_lion(terminals, max_depth, depth + 1) for _ in range(arity)]
    return Node(operation, children)


@dataclass
class Context:
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


def update_ctxs(
    positions: np.ndarray, previous_ctxs: list[Context] | None
) -> list[Context]:
    """generate spatial contexts for each lion

    Args:
        positions np.ndarray: shape (5, 2), gazelle=0, then each lion indexed by number
        ctxs np.ndarray: shape (5, 2), gazelle (dummy)=0, then each lion indexed by number


    Returns:
        list[dict]: list of contexts, indexed by lion number (0 is a dummy for gazelle)
    """

    gazelle = positions[0]
    vecs_to_gazelle = np.array(
        [vector_between(lion, gazelle) for lion in positions[1:]]
    )
    dist_to_gazelle = np.linalg.norm(vecs_to_gazelle, axis=1)
    nearest = np.argmin(dist_to_gazelle)
    caught_gazelle = True if dist_to_gazelle[nearest] < 1 else False
    nearest = vecs_to_gazelle[nearest]

    norms = np.zeros((5, 5))
    angles = np.zeros((5, 5))
    vecs = np.zeros((5, 5, 2))
    for a, b in combinations([1, 2, 3, 4], 2):
        vec = vector_between(positions[a], positions[b])
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
            else np.random.random_sample(2) * np.array([c.WIDTH, c.HEIGHT])
        )
        closest_lion = vecs[lion_idx, 1 + np.argmin(norms[lion_idx, 1:])]

        heading_angle = np.arctan2(last[1], last[0])
        other_idxs = [other for other in range(1, 5) if other != lion_idx]
        rel_angles = np.array(
            [angle_diff(angles[lion_idx, other], heading_angle) for other in other_idxs]
        )
        rlion = vecs[lion_idx, other_idxs[np.argmin(rel_angles)]]
        llion = vecs[lion_idx, other_idxs[np.argmax(rel_angles)]]

        ctx = Context(
            last=last,
            gazelle=vecs_to_gazelle[lion_idx - 1],
            vecs_to_gazelle=vecs_to_gazelle,
            dist_to_gazelle=dist_to_gazelle,
            nearest=nearest,
            caught_gazelle=caught_gazelle,
            lion=closest_lion,
            rlion=rlion,
            llion=llion,
            lion1=positions[1],
            lion2=positions[2],
            lion3=positions[3],
            lion4=positions[4],
        )
        ctxs.append(ctx)

    return ctxs


# ---------------------------------------------------------------------------
# Copy, crossover, mutation
# ---------------------------------------------------------------------------


def copy_tree(node: Node) -> Node:
    return Node(node.operation, [copy_tree(c) for c in node.children])


def _all_nodes(node: Node) -> list[Node]:
    result = [node]
    for child in node.children:
        result.extend(_all_nodes(child))
    return result


def overwrite_node(target: Node, source: Node) -> None:
    """Overwrite target in-place with source contents."""
    target.operation = source.operation
    target.children = source.children


def crossover(lion_a: Node, lion_b: Node) -> Node:
    """Swap a random subtree from lion_b into a random point in a copy of lion_a.
    Falls back to lion_a copy if result exceeds MAX_SIZE."""
    child = copy_tree(lion_a)
    point_a = random.choice(_all_nodes(child))
    subtree_b = copy_tree(random.choice(_all_nodes(lion_b)))
    overwrite_node(point_a, subtree_b)
    if child.size() > c.MAX_SIZE:
        return copy_tree(lion_a)
    return child


def mutate(node: Node, terminals: list[str]) -> Node:
    """Replace a random subtree with a freshly grown one."""
    child = copy_tree(node)
    point = random.choice(_all_nodes(child))
    new_subtree = random_lion(terminals, max_depth=4)
    overwrite_node(point, new_subtree)
    return child


class Pride:
    """A team of 4 lions. Each lion has its own GP tree.
    For clones, all four trees are identical.
    For free/restricted breeding, each is distinct."""

    def __init__(self, lions: list[Node]):
        assert len(lions) == 4
        self.lions = lions

    @classmethod
    def random(cls, terminals: list[str], breeding_strategy: str) -> "Pride":
        if breeding_strategy == "clone":
            t = random_lion(terminals)
            return cls([copy_tree(t) for _ in range(4)])
        return cls([random_lion(terminals) for _ in range(4)])

    def get_lion_vectors(self, ctxs: list[Context]) -> np.ndarray:
        """ctxs[i] is the sensor context for lion i. Returns movement vectors."""
        vectors = np.array(
            [lion.process_inputs(ctx) for lion, ctx in zip(self.lions, ctxs[1:])]
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms


def get_gazelle_vector(ctxs: list[Context]) -> np.ndarray:
    vecs_to_gazelle = ctxs[1].vecs_to_gazelle
    dists_to_gazelle = ctxs[1].dist_to_gazelle
    importance_valued_vecs = c.MAX_NORM - dists_to_gazelle
    fear_vecs = (vecs_to_gazelle / dists_to_gazelle[:, None]) * importance_valued_vecs[
        :, None
    ]
    result = np.sum(fear_vecs, axis=0)
    return result / np.linalg.norm(result) * c.GAZELLE_STEP


def toroidal_step(positions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    positions = positions + vectors
    return positions % c.WORLD


class Run:
    def __init__(self, terminals, breeding_strategy: str) -> None:
        self.population = [
            Pride.random(terminals, breeding_strategy)
            for _ in range(c.POPULATION_COUNT)
        ]
        self.loss = np.zeros((c.GEN_COUNT, c.POPULATION_COUNT))
        self.breeding_strategy = breeding_strategy
        self.terminals = terminals
        self.best_loss = np.zeros(c.GEN_COUNT)
        self.best_pride = []
        self.avg_loss = np.zeros(c.GEN_COUNT)
        self.best_positions = np.zeros(
            (c.GEN_COUNT, c.STEPS_PER_SIM, 5, 2)
        )  # [gen, steps, agents, xy]

    def select_parents(self, gen):
        parents = []
        while len(parents) < c.PARENT_COUNT:
            idxs = random.sample(range(c.POPULATION_COUNT), c.TOURNAMENT_N)
            parent_idx = idxs[np.argmin(self.loss[gen, idxs])]
            parents.append(self.population[parent_idx])
        return parents

    def breed(self, parents):
        children = []

        ## crossover
        for i in range(c.CROSSOVER_COUNT):
            i *= 2
            match self.breeding_strategy:
                case "free":
                    j = random.choice([0, 1, 2, 3])
                    k = random.choice([0, 1, 2, 3])
                case "restricted":
                    j = k = random.choice([0, 1, 2, 3])
                case _:
                    j = k = 0
            child = Pride([copy_tree(lion) for lion in parents[i].lions])
            node_a = child.lions[j]
            node_b = parents[i + 1].lions[k]
            child.lions[j] = crossover(node_a, node_b)
            if self.breeding_strategy == "clone":
                child = Pride([copy_tree(child.lions[0]) for _ in range(4)])
            children.append(child)

        ## mutation
        for i in range(c.CROSSOVER_COUNT, c.POPULATION_COUNT):
            j = random.choice([0, 1, 2, 3])
            child = Pride([copy_tree(lion) for lion in parents[i].lions])
            child.lions[j] = mutate(child.lions[j], self.terminals)
            children.append(child)

        assert len(children) == c.POPULATION_COUNT
        return children

    def run_gens(self):
        for gen in range(c.GEN_COUNT):
            caught_count = 0
            for idx, pride in enumerate(self.population):
                loss, count = self.run_sims(pride)
                caught_count += count
                self.loss[gen, idx] = loss
            print(
                f"Generation {gen}, caught={caught_count}, loss={np.mean(self.loss[gen, :]):.4f}"
            )
            parents = self.select_parents(gen)
            self.population = self.breed(parents)
            best_pride = np.argmin(self.loss[gen])

            self.best_pride.append(self.population[best_pride])
            self.best_loss[gen] = self.loss[gen, best_pride]
            self.avg_loss[gen] = np.mean(self.loss[gen])
            positions, ctxs, _ = self.single_simulation(self.population[best_pride])
            self.best_positions[gen] = positions

    def run_sims(self, pride: Pride):
        loss = 0
        caught_count = 0
        for _ in range(c.SIMS_PER_GEN):
            positions, ctxs, caught = self.single_simulation(pride)
            if caught:
                caught_count += 1
            else:
                loss += np.linalg.norm(ctxs[1].nearest) - 1
        return loss / c.SIMS_PER_GEN, caught_count

    def single_simulation(self, pride: Pride) -> tuple[np.ndarray, list[Context], bool]:
        positions = np.zeros(shape=(c.STEPS_PER_SIM, 5, 2))
        positions[0] = random_positions()
        ctxs = update_ctxs(positions[0], None)
        for step in range(1, c.STEPS_PER_SIM):
            lion_vecs = pride.get_lion_vectors(ctxs)
            for i, vec in enumerate(lion_vecs):
                ctxs[i + 1].heading = vec
            gazelle_vec = get_gazelle_vector(ctxs)
            vectors = np.array([gazelle_vec, *lion_vecs])
            positions[step] = toroidal_step(positions[step - 1], vectors)
            ctxs = update_ctxs(positions[step], ctxs)
            if ctxs[1].caught_gazelle:
                # print(
                #     f"step {step} caught: {ctxs[1].caught_gazelle}, nearest dist: {np.linalg.norm(ctxs[1].nearest)}"
                # )
                return positions, ctxs, True
        return positions, ctxs, False

    def save_sim(self, idx, title="", save_path="data"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(save_path) / title / str(idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{timestamp}.npy"
        np.save(path, self.best_positions[idx])
        return path


def random_positions():
    while True:
        positions = np.random.random_sample(size=(5, 2)) * np.array([c.WIDTH, c.HEIGHT])
        dists = [np.linalg.norm(positions[i + 1] - positions[0]) for i in range(4)]
        if min(dists) > 1:
            return positions


if __name__ == "__main__":
    r = Run(NAME_TERMINALS, "restricted")
    r.run_gens()
    p1 = r.save_sim(idx=0)
    p2 = r.save_sim(idx=c.GEN_COUNT - 1)
    viz_sim(p1, title="og")
    viz_sim(p2, title="final")
