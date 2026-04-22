## std lib
from pathlib import Path
from itertools import product
import json
from ast import literal_eval
import pprint


## 3rd p lib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

## project
from src.node import INTERNALS
from src.run import TERMINAL_DICT, BASE_TERMINALS, DEICTIC_TERMINALS, NAME_TERMINALS
from src.config_parser import Config, load_config


# constants
IMG_WIDTH = 12
IMG_HEIGHT = 6
plt.style.use(["ggplot"])


def build_counts(max_depth, terminals_used):
    counts = {
        operator: [
            0 for _ in range(max_depth + 1)
        ]  #  TODO: Change once rerun with max_depth bug fixed
        for operator in terminals_used + INTERNALS
    }
    return counts


def count_operations(node, terminals_used, level=0, counts=None, max_depth=17):
    if counts is None:
        counts = build_counts(max_depth, terminals_used)
    op = node["op"]
    counts[op][level] += 1
    for child in node["children"]:
        count_operations(child, terminals_used, level + 1, counts, max_depth=max_depth)
    return counts


def get_gen_op_counts(file_path, gen_idx, max_depth, terminals_used):
    with open(file_path) as f:
        for i, line in enumerate(f):
            if i == gen_idx:
                population = json.loads(line)
                break

    counts = build_counts(max_depth, terminals_used)
    for pride in population:
        for lion in pride:
            count_operations(lion, terminals_used, 0, counts)
    df = pd.DataFrame.from_dict(counts, orient="index")
    return df


def plot_depths(gens, dfs, ax, title_specs):
    depth_profiles = pd.DataFrame(
        {f"Gen {gen}": df.sum(axis=0) for gen, df in zip(gens, dfs)}
    )
    depth_profiles.plot(ax=ax)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Node count")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Depth profile — {title_specs}")


def plot_heatmaps(file_path, gens, title_spec, terminal, max_depth=17):
    terminals_used = TERMINAL_DICT[terminal]

    dfs = [get_gen_op_counts(file_path, gen, max_depth, terminals_used) for gen in gens]
    n_axes = len(gens) + 1
    n_rows = (n_axes + 1) // 2

    fig, axes = plt.subplots(n_rows, 2, figsize=(IMG_WIDTH, IMG_HEIGHT * n_rows))
    axes = axes.flatten()

    for i, (ax, df, title) in enumerate(
        zip(axes, dfs, [f"Generation {gen}" for gen in gens])
    ):
        df = df.div(df.sum())  ## normalize by column (depth)
        sns.heatmap(df, ax=ax, cmap="viridis", annot=False, cbar=True)
        ax.set_title(title)
        ax.set_xlabel("Node Depth")

    plot_depths(gens, dfs, axes[len(gens)], title_spec)
    plt.suptitle(f"Heatplots for {title_spec}, normalized by depth")
    plt.tight_layout()
    plt.show()
