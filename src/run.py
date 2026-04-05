"""
**Author**:
    Fitz Koch
**Created**:
    2026-04-01
**Description**:
    Code written by Fitz Koch with some bug checking from Claude
"""

import random
from pathlib import Path
from datetime import datetime
import json

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

import math_helpers as m
from config_parser import Config
from node import Pride, copy_tree, crossover, mutate, get_gazelle_vector
from context import Context, update_ctxs, update_gazelle_ctx


# Terminal sets per sensing mode
BASE_TERMINALS = ["last", "rand-dir", "gazelle"]
DEICTIC_TERMINALS = BASE_TERMINALS + ["nearest", "lion", "rlion", "llion"]
NAME_TERMINALS = BASE_TERMINALS + ["lion-1", "lion-2", "lion-3", "lion-4"]

TERMINAL_DICT = {
    "Base": BASE_TERMINALS,
    "Deitic": DEICTIC_TERMINALS,
    "Name": NAME_TERMINALS,
}


def toroidal_step(positions: np.ndarray, vectors: np.ndarray, c: Config) -> np.ndarray:
    positions = positions + vectors
    return positions % c.world


class Run:
    def __init__(
        self, terminal_type, breeding_strategy: str, logger, c: Config
    ) -> None:
        ## saving inputs and documenting
        self.logger = logger
        self.c = c
        timestamp = datetime.now()
        self.timestamp = timestamp.strftime("%Y-%m-%d")
        self.logger.info(
            f"Starting {self.c.title}: {breeding_strategy}, {terminal_type} "
        )
        self.terminal_type = terminal_type
        self.terminals = TERMINAL_DICT[terminal_type]

        #
        self.population = [
            Pride.random(self.terminals, breeding_strategy, self.logger, c)
            for _ in range(c.population_count)
        ]
        self.loss = np.zeros((c.gen_count, c.population_count))
        self.breeding_strategy = breeding_strategy
        self.terminals = self.terminals
        self.best_loss = np.zeros(c.gen_count)
        self.best_pride = []
        self.avg_loss = np.zeros(c.gen_count)
        self.best_positions = np.zeros((c.gen_count, c.steps_per_sim * 2 + 1, 5, 2))
        self.random_positions = np.zeros((c.gen_count, c.steps_per_sim * 2 + 1, 5, 2))

        self.pride_history = [pride.to_dict() for pride in self.population]

    def select_parents(self, gen):
        parents = []
        while len(parents) < self.c.parent_count:
            idxs = random.sample(range(self.c.population_count), self.c.tournament_n)
            parent_idx = idxs[np.argmin(self.loss[gen, idxs])]
            parents.append(self.population[parent_idx])
        return parents

    def breed(self, parents):
        ## could parallelize but doesn't seem like a big time sink
        children = []

        ## crossover
        for i in range(self.c.crossover_count):
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
            child.lions[j] = crossover(node_a, node_b, self.c)
            if self.breeding_strategy == "Clone":
                child = Pride(
                    [copy_tree(child.lions[0]) for _ in range(4)], self.logger
                )
            children.append(child)

        ## mutation
        for i in range(
            self.c.crossover_count, self.c.crossover_count + self.c.mutation_count
        ):
            j = random.choice([0, 1, 2, 3])
            child = Pride([copy_tree(lion) for lion in parents[i].lions], self.logger)
            child.lions[j] = mutate(child.lions[j], self.terminals, self.c)
            if self.breeding_strategy == "Clone":
                child = Pride(
                    [copy_tree(child.lions[j]) for _ in range(4)], self.logger
                )
            children.append(child)

        ## reproduction
        for i in range(
            self.c.crossover_count + self.c.mutation_count, self.c.population_count
        ):
            child = parents[i]
            children.append(child)

        assert len(children) == self.c.population_count
        self.pride_history.append([pride.to_dict() for pride in children])
        return children

    def run_gens(self):
        run_sims_c = partial(run_sims, c=self.c)
        with ProcessPoolExecutor() as executor:
            for gen in range(self.c.gen_count):
                caught_count = 0
                results = list(executor.map(run_sims_c, self.population))
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
                positions, _, _ = single_simulation(
                    self.population[best_pride_idx], self.c
                )
                self.best_positions[gen] = positions

                random_pride_idx = np.random.randint(self.c.population_count)
                positions, _, _ = single_simulation(
                    self.population[random_pride_idx], self.c
                )
                self.random_positions[gen] = positions

                self.avg_loss[gen] = np.mean(self.loss[gen])

                ## modification
                parents = self.select_parents(gen)
                self.population = self.breed(parents)

    def save_positions(self, save_path="results"):
        out_dir = (
            Path(save_path) / self.c.title / self.terminal_type / self.breeding_strategy
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "best_positions.npy"
        np.save(path, self.best_positions)
        self.logger.info(f"Saved best positions to {path}")

        path = out_dir / "random_positions.npy"
        np.save(path, self.random_positions)
        self.logger.info(f"Saved random positions to {path}")
        return path

    def save_run_results(self, save_path="results"):
        out_dir = (
            Path(save_path) / self.c.title / self.terminal_type / self.breeding_strategy
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "loss_results.npz"
        np.savez(
            save_path,
            best_loss=self.best_loss,
            avg_loss=self.avg_loss,
        )
        self.logger.info(f"Saved average and best loss to {save_path} ")

    def save_pride_history(self, save_path="results"):
        out_dir = (
            Path(save_path) / self.c.title / self.terminal_type / self.breeding_strategy
        )
        history_path = out_dir / "pride_history.json"
        with open(history_path, "w") as f:
            json.dump(self.pride_history, f)


def random_positions(c: Config):
    while True:
        positions = np.random.random_sample(size=(5, 2)) * np.array([c.width, c.height])
        dists = [
            np.linalg.norm(m.vector_between(positions[i + 1], positions[0], c))
            for i in range(4)
        ]
        if min(dists) > c.initial_dist:
            return positions


def run_sims(pride: Pride, c: Config):
    """Run multiple simulations. Outside `Run` class so that it doesn't require pickling the whole thing to run in parallel.

    Args:
        pride (Pride): _description_
        c (Config): _description_

    Returns:
        _type_: _description_
    """
    loss = 0
    caught_count = 0
    for _ in range(c.sims_per_gen):
        _, ctxs, caught = single_simulation(pride, c)
        if caught:
            caught_count += 1
        else:
            loss += np.linalg.norm(ctxs[1].nearest) - 1
    return loss / c.sims_per_gen, caught_count


def single_simulation(
    pride: Pride, c: Config
) -> tuple[np.ndarray, list[Context], bool]:
    """Run a single simulation. Outside `Run` class so that it doesn't require pickling the whole thing to run in parallel.

    Args:
        pride (Pride): _description_
        c (Config): _description_

    Returns:
        tuple[np.ndarray, list[Context], bool]: _description_
    """
    positions = np.zeros(shape=(c.steps_per_sim * 2 + 1, 5, 2))
    positions[0] = random_positions(c)
    ctxs = update_ctxs(positions[0], None, c)
    for step in range(1, c.steps_per_sim * 2 + 1):
        if step % 2:
            vectors = np.zeros((5, 2))
            vectors[0] = get_gazelle_vector(ctxs, c)
            positions[step] = toroidal_step(positions[step - 1], vectors, c)
            ctxs = update_gazelle_ctx(positions[step], ctxs, c)

        else:
            lion_vecs = pride.get_lion_vectors(ctxs)
            for i, vec in enumerate(lion_vecs):
                ctxs[i + 1].heading = vec
            vectors = np.array([np.zeros(2), *lion_vecs])
            positions[step] = toroidal_step(positions[step - 1], vectors, c)
            ctxs = update_ctxs(positions[step], ctxs, c)
            if ctxs[1].caught_gazelle:
                return positions, ctxs, True
    return positions, ctxs, False
