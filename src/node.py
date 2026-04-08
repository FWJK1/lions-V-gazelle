from typing import Callable
from . import math_helpers as m
import numpy as np
import random
from .config_parser import Config

from .context import Context


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
    "inv": (1, lambda ctx, a: m.invert(a, ctx.config)),
    "ifdot": (4, lambda ctx, a, b, c, d: c if np.dot(a, b) >= 0 else d),
    "if>=": (
        4,
        lambda ctx, a, b, c, d: c if np.linalg.norm(a) >= np.linalg.norm(b) else d,
    ),
}
# fmt: on
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
        if not np.all(np.isfinite(vector)):
            return np.zeros(2)
        return vector

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def __repr__(self) -> str:
        return self._fmt()

    def to_dict(self) -> dict:
        return {"op": self.operation, "children": [c.to_dict() for c in self.children]}

    def _fmt(self, indent: int = 0) -> str:
        prefix = "  " * indent
        if not self.children:
            return f"{prefix}{self.operation}"
        lines = [f"{prefix}{self.operation}"]
        for child in self.children:
            lines.append(child._fmt(indent + 1))
        return "\n".join(lines)


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


def crossover(lion_a: Node, lion_b: Node, c: Config, count=0) -> Node:
    """Swap a random subtree from lion_b into a random point in a copy of lion_a.
    Recurses if result exceeds MAX_SIZE."""
    child = copy_tree(lion_a)
    point_a = random.choice(_all_nodes(child))
    subtree_b = copy_tree(random.choice(_all_nodes(lion_b)))
    overwrite_node(point_a, subtree_b)
    if child.size() > c.max_size or child.depth() > c.max_depth:
        if count < 5:
            return crossover(lion_a, lion_b, c, count + 1)
        return copy_tree(lion_a)
    return child


def mutate(node: Node, terminals: list[str], c: Config) -> Node:
    """Replace a random subtree with a freshly grown one."""
    child = copy_tree(node)
    point = random.choice(_all_nodes(child))
    new_subtree = random_lion(terminals, c, max_depth=4)
    overwrite_node(point, new_subtree)
    return child


def random_lion(terminals, c: Config, max_depth=17, depth=0, root=False) -> Node:
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
        depth > 0 and np.random.random_sample() < c.early_terminal_probability
    )

    if pick_terminal:
        return Node(random.choice(terminals), root=root)

    operation = random.choice(INTERNALS)
    arity, _ = OPERATORS[operation]
    children = [
        random_lion(terminals, c, max_depth, depth + 1, root=False)
        for _ in range(arity)
    ]
    return Node(operation, children, root=root)


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

    def to_dict(self) -> list:
        return [lion.to_dict() for lion in self.lions]

    @classmethod
    def random(
        cls, terminals: list[str], breeding_strategy: str, logger, c: Config
    ) -> "Pride":
        if breeding_strategy == "Clone":
            t = random_lion(terminals, c, root=True)
            return cls([copy_tree(t) for _ in range(4)], logger)
        return cls([random_lion(terminals, c, root=True) for _ in range(4)], logger)

    def get_lion_vectors(self, ctxs: list[Context]) -> np.ndarray:
        """ctxs[i] is the sensor context for lion i. Returns movement vectors."""
        vectors = np.array(
            [lion.process_inputs(ctx) for lion, ctx in zip(self.lions, ctxs[1:])]
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        zero_mask = (norms == 0).flatten()
        if np.any(zero_mask):
            for i in np.where(zero_mask)[0]:
                vectors[i] = m.random_unit_vector()
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


def get_gazelle_vector(ctxs: list[Context], c: Config) -> np.ndarray:
    vecs_to_gazelle = ctxs[1].vecs_to_gazelle
    dists_to_gazelle = ctxs[1].dist_to_gazelle
    importance_valued_vecs = c.max_norm - dists_to_gazelle
    fear_vecs = (vecs_to_gazelle / dists_to_gazelle[:, None]) * importance_valued_vecs[
        :, None
    ]
    result = np.sum(fear_vecs, axis=0)
    return result / np.linalg.norm(result) * c.gazelle_step
