"""
**Author**:
    Fitz Koch
**Created**:
    2026-04-05
**Description**:
    Parse configs into dataclass for accessing.
"""

from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Config:
    # [overview]
    title: str
    terminals: list[str]
    strategies: list[str]

    # [world]
    width: int
    height: int

    # [animals]
    gazelle_step: int
    initial_dist: int

    # [gp_trees]
    max_depth: int
    max_size: int
    early_terminal_probability: float

    # [run_counts]
    gen_count: int
    steps_per_sim: int
    sims_per_gen: int

    # [population]
    population_count: int
    mutation_percentage: float
    reproduction_percentage: float
    tournament_n: int

    # [derived]
    max_norm: float = field(init=False)
    world: np.ndarray = field(init=False)
    crossover_percentage: float = field(init=False)

    mutation_count: int = field(init=False)
    reproduction_count: int = field(init=False)
    crossover_count: int = field(init=False)
    parent_count: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self, "max_norm", np.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)
        )
        object.__setattr__(
            self, "world", np.array([self.width, self.height], dtype=float)
        )
        object.__setattr__(
            self,
            "crossover_percentage",
            1 - self.mutation_percentage - self.reproduction_percentage,
        )
        object.__setattr__(
            self,
            "mutation_count",
            int(self.population_count * self.mutation_percentage),
        )
        object.__setattr__(
            self,
            "reproduction_count",
            int(self.population_count * self.reproduction_percentage),
        )
        object.__setattr__(
            self,
            "crossover_count",
            int(self.population_count * self.crossover_percentage),
        )
        object.__setattr__(
            self,
            "parent_count",
            self.mutation_count + self.reproduction_count + 2 * self.crossover_count,
        )


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    flat = {k: v for section in raw.values() for k, v in section.items()}
    return Config(**flat)
