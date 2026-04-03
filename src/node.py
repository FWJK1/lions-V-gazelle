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
from itertools import combinations, product
from pathlib import Path
from datetime import datetime
from typing import Callable

from concurrent.futures import ProcessPoolExecutor

import numpy as np

import constants as c
import math_helpers as m
from logging_util import set_up_logging

OPERATORS: dict[str, tuple[int, Callable]] = {
    # --- Terminals (arity 0): look up pre-computed value in ctx ---
    # standard
    "last": (0, lambda ctx: ctx.last),
    "rand-dir": (0, lambda ctx: m.random_unit_vector()),
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
    "->90": (1, lambda ctx, a: m.rotate_90(a)),
    "rand": (1, lambda ctx, a: m.rand_scale(a)),
    "inv": (1, lambda ctx, a: m.invert(a)),
    "ifdot": (4, lambda ctx, a, b, c, d: c if np.dot(a, b) >= 0 else d),
    "if>=": (
        4,
        lambda ctx, a, b, c, d: c if np.linalg.norm(a) >= np.linalg.norm(b) else d,
    ),
}
# fmt: on


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


# Terminal sets per sensing mode
BASE_TERMINALS = ["last", "rand-dir", "gazelle"]
DEICTIC_TERMINALS = BASE_TERMINALS + ["nearest", "lion", "rlion", "llion"]
NAME_TERMINALS = BASE_TERMINALS + ["lion-1", "lion-2", "lion-3", "lion-4"]

TERMINAL_DICT = {
    "Base": BASE_TERMINALS,
    "Deitic": DEICTIC_TERMINALS,
    "Name": NAME_TERMINALS,
}


INTERNALS = ["+", "-", "*2", "/2", "->90", "rand", "inv", "ifdot", "if>="]


class Node:
    def __init__(
        self, operation: str, children: list["Node"] | None = None, root=False
    ) -> None:
        self.operation = operation
        self.children = children or []
        self.root = root

    def process_inputs(self, ctx: Context) -> np.ndarray:
        arity, func = OPERATORS[self.operation]
        child_vals = [child.process_inputs(ctx) for child in self.children]
        vector = func(ctx, *child_vals)
        if np.all(vector == 0) or not np.all(np.isfinite(vector)):
            return m.random_unit_vector()
        return vector

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
        return f"{'root' if self.root else 'node'}:  {self.operation} on {[c for c in child_vals]}"


def random_lion(terminals, max_depth=c.MAX_DEPTH, depth=0, root=False) -> Node:
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
        return Node(random.choice(terminals), root=root)

    operation = random.choice(INTERNALS)
    arity, _ = OPERATORS[operation]
    children = [
        random_lion(terminals, max_depth, depth + 1, root=False) for _ in range(arity)
    ]
    return Node(operation, children, root=root)


def get_gazelle_vectors(positions: np.ndarray) -> tuple:
    gazelle = positions[0]
    vecs_to_gazelle = np.array(
        [m.vector_between(lion, gazelle) for lion in positions[1:]]
    )
    dist_to_gazelle = np.linalg.norm(vecs_to_gazelle, axis=1)
    nearest = np.argmin(dist_to_gazelle)
    caught_gazelle = True if dist_to_gazelle[nearest] < 1 else False
    nearest = vecs_to_gazelle[nearest]
    return vecs_to_gazelle, dist_to_gazelle, nearest, caught_gazelle


def update_gazelle_ctx(
    positions: np.ndarray, previous_ctxs: list[Context]
) -> list[Context]:
    """generate spatial contexts for the gazelle, which is shared.

    Args:
        positions (np.ndarray): _description_
        previous_ctxs (list[Context]): _description_

    Returns:
        list[Context]: _description_
    """
    vecs_to_gazelle, dist_to_gazelle, nearest, caught_gazelle = get_gazelle_vectors(
        positions
    )
    ctxs = []
    ctxs.append(None)
    for lion_idx in range(1, 5):
        ctx = Context(
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
    positions: np.ndarray, previous_ctxs: list[Context] | None
) -> list[Context]:
    """generate spatial contexts for each lion and the gazelle

    Args:
        positions np.ndarray: shape (5, 2), gazelle=0, then each lion indexed by number
        ctxs np.ndarray: shape (5, 2), gazelle (dummy)=0, then each lion indexed by number


    Returns:
        list[dict]: list of contexts, indexed by lion number (0 is a dummy for gazelle)
    """

    vecs_to_gazelle, dist_to_gazelle, nearest, caught_gazelle = get_gazelle_vectors(
        positions
    )

    # set defaults to zero
    norms = np.zeros((5, 5))
    angles = np.zeros((5, 5))
    vecs = np.zeros((5, 5, 2))
    for a, b in combinations([1, 2, 3, 4], 2):
        vec = m.vector_between(positions[a], positions[b])
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
    For Clones, all four trees are identical.
    For Free/Restricted breeding, each is distinct."""

    def __init__(self, lions: list[Node], logger):
        assert len(lions) == 4
        self.logger = logger
        self.lions = lions

    def __repr__(self) -> str:
        return "\n".join([repr(lion) for lion in self.lions])

    @classmethod
    def random(cls, terminals: list[str], breeding_strategy: str, logger) -> "Pride":
        if breeding_strategy == "Clone":
            t = random_lion(terminals, root=True)
            return cls([copy_tree(t) for _ in range(4)], logger)
        return cls([random_lion(terminals, root=True) for _ in range(4)], logger)

    def get_lion_vectors(self, ctxs: list[Context]) -> np.ndarray:
        """ctxs[i] is the sensor context for lion i. Returns movement vectors."""
        vectors = np.array(
            [lion.process_inputs(ctx) for lion, ctx in zip(self.lions, ctxs[1:])]
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            self.logger.debug(norms)
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
    def __init__(self, terminal_type, breeding_strategy: str, logger) -> None:
        ## saving inputs and documenting
        self.logger = logger
        timestamp = datetime.now()
        self.timestamp = timestamp.strftime("%Y-%m-%d")
        self.logger.info(
            f"Starting {breeding_strategy}, {terminal_type} at {timestamp.strftime('%Y-%m-%d_%H-%M')} "
        )
        self.terminal_type = terminal_type
        self.terminals = TERMINAL_DICT[terminal_type]

        #
        self.population = [
            Pride.random(self.terminals, breeding_strategy, self.logger)
            for _ in range(c.POPULATION_COUNT)
        ]
        self.loss = np.zeros((c.GEN_COUNT, c.POPULATION_COUNT))
        self.breeding_strategy = breeding_strategy
        self.terminals = self.terminals
        self.best_loss = np.zeros(c.GEN_COUNT)
        self.best_pride = []
        self.avg_loss = np.zeros(c.GEN_COUNT)
        self.best_positions = np.zeros(
            (c.GEN_COUNT, c.STEPS_PER_SIM * 2 + 1, 5, 2)
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
                case "Free":
                    j = random.choice([0, 1, 2, 3])
                    k = random.choice([0, 1, 2, 3])
                case "Restricted":
                    j = k = random.choice([0, 1, 2, 3])
                case _:
                    j = k = 0
            child = Pride([copy_tree(lion) for lion in parents[i].lions], self.logger)
            node_a = child.lions[j]
            node_b = parents[i + 1].lions[k]
            child.lions[j] = crossover(node_a, node_b)
            if self.breeding_strategy == "Clone":
                child = Pride(
                    [copy_tree(child.lions[0]) for _ in range(4)], self.logger
                )
            children.append(child)

        ## mutation
        for i in range(c.CROSSOVER_COUNT, c.POPULATION_COUNT):
            j = random.choice([0, 1, 2, 3])
            child = Pride([copy_tree(lion) for lion in parents[i].lions], self.logger)
            child.lions[j] = mutate(child.lions[j], self.terminals)
            if self.breeding_strategy == "Clone":
                child = Pride(
                    [copy_tree(child.lions[j]) for _ in range(4)], self.logger
                )
            children.append(child)

        assert len(children) == c.POPULATION_COUNT
        return children

    def run_gens(self):
        with ProcessPoolExecutor() as executor:
            for gen in range(c.GEN_COUNT):
                caught_count = 0
                results = list(executor.map(run_sims, self.population))
                for idx, (loss, count) in enumerate(results):
                    caught_count += count
                    self.loss[gen, idx] = loss

                self.logger.info(
                    f"Strat= {self.breeding_strategy}, Term={self.terminal_type}, Generation {gen}, caught={caught_count}, avg_loss={np.mean(self.loss[gen, :]):.4f}, best_loss={np.min(self.loss[gen, :]):.4f}"
                )

                ## storage
                best_pride_idx = np.argmin(self.loss[gen])
                best_pride = self.population[best_pride_idx]
                self.logger.debug(repr(best_pride))
                self.best_pride.append(best_pride)
                self.best_loss[gen] = self.loss[gen, best_pride_idx]
                self.avg_loss[gen] = np.mean(self.loss[gen])
                positions, _, _ = single_simulation(self.population[best_pride_idx])
                self.best_positions[gen] = positions

                ## modification
                parents = self.select_parents(gen)
                self.population = self.breed(parents)

    def save_best_positions(self, save_path="position_data"):
        out_dir = Path(save_path) / self.terminal_type / self.breeding_strategy
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.timestamp}.npy"
        np.save(path, self.best_positions)
        return path

    def save_run_results(self, save_path="results"):
        out_dir = Path(save_path) / self.terminal_type / self.breeding_strategy
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.timestamp}.npy"
        np.savez(
            out_dir / f"{self.timestamp}_checkpoint.npz",
            best_loss=self.best_loss,
            avg_loss=self.avg_loss,
        )
        self.logger.info(
            f"Saved average and best loss to {out_dir / f'{self.timestamp}_checkpoint.npz'} "
        )


def random_positions():
    while True:
        positions = np.random.random_sample(size=(5, 2)) * np.array([c.WIDTH, c.HEIGHT])
        dists = [
            np.linalg.norm(m.vector_between(positions[i + 1], positions[0]))
            for i in range(4)
        ]
        if min(dists) > 1:
            return positions


def run_sims(pride: Pride):
    """Run multiple simulations. Outside `Run` class so that it doesn't require pickling the whole thing to run in parallel.

    Args:
        pride (Pride): _description_

    Returns:
        _type_: _description_
    """
    loss = 0
    caught_count = 0
    for _ in range(c.SIMS_PER_GEN):
        _, ctxs, caught = single_simulation(pride)
        if caught:
            caught_count += 1
        else:
            loss += np.linalg.norm(ctxs[1].nearest) - 1
    return loss / c.SIMS_PER_GEN, caught_count


def single_simulation(pride: Pride) -> tuple[np.ndarray, list[Context], bool]:
    """Run a single simulation. Outside `Run` class so that it doesn't require pickling the whole thing to run in parallel.

    Args:
        pride (Pride): _description_

    Returns:
        tuple[np.ndarray, list[Context], bool]: _description_
    """
    positions = np.zeros(shape=(c.STEPS_PER_SIM * 2 + 1, 5, 2))
    positions[0] = random_positions()
    ctxs = update_ctxs(positions[0], None)
    for step in range(1, c.STEPS_PER_SIM * 2 + 1):
        if step % 2:
            vectors = np.zeros((5, 2))
            vectors[0] = get_gazelle_vector(ctxs)
            positions[step] = toroidal_step(positions[step - 1], vectors)
            ctxs = update_gazelle_ctx(positions[step], ctxs)

        else:
            lion_vecs = pride.get_lion_vectors(ctxs)
            for i, vec in enumerate(lion_vecs):
                ctxs[i + 1].heading = vec
            vectors = np.array([np.zeros(2), *lion_vecs])
            positions[step] = toroidal_step(positions[step - 1], vectors)
            ctxs = update_ctxs(positions[step], ctxs)
            if ctxs[1].caught_gazelle:
                return positions, ctxs, True
    return positions, ctxs, False


if __name__ == "__main__":
    sensing_terminals = ["Name", "Deitic", "Base"]
    breeding_strategies = ["Restricted", "Clone", "Free"]
    logger = set_up_logging()
    for terminal, strat in product(sensing_terminals, breeding_strategies):
        run = Run(terminal, strat, logger)
        run.run_gens()
        run.save_best_positions()
        run.save_run_results()
