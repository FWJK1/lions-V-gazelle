# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
import sys, os
from pathlib import Path

_project_root = Path.cwd()
while not (_project_root / "src").exists():
    _project_root = _project_root.parent
os.chdir(_project_root)
#
#
#
#
#
# | cache: false

## std lib
from pathlib import Path
from itertools import product

## 3rd p lib
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# project code
from src.config_parser import load_config
from src.viz import viz_sim, plot_loss_curves



# constants
IMG_WIDTH = 12
plt.style.use(["ggplot"])
#
#
#
#
#
#| tags: [parameters]

experiment = "Default"
#
#
#
config_path = Path("configs") / f"{experiment}.toml"
config = load_config(config_path)
print(config)
#
#
#
#
sensing_terminals = ["Name", "Deitic", "Base"]
path_end = "loss_results.npz"
result_path = Path("results")

for terminal in sensing_terminals:
    clone_data = np.load(result_path / experiment / terminal / "Clone" / path_end)
    free_data = np.load(result_path / experiment / terminal / "Free" / path_end)
    restricted_data = np.load(result_path / experiment / terminal / "Restricted" / path_end)  # fmt: skip

    plot_loss_curves(
        clone_data, free_data, restricted_data, title=f"{experiment}: {terminal}"
    )
#
#
#
#
#
#
anim = Path("animations")
terminals = ["Base", "Deitic", "Name"]
strategies = ["Clone", "Free", "Restricted"]
qualities = ["best_positions", "random_positions"]
generations = ["first", "final"]


for terminal, strategy, quality, generation in product(
    terminals, strategies, qualities, generations
):
    path = anim / experiment / terminal / strategy / f"{quality}_{generation}.html"
    with open(path) as f:
        display(HTML(f.read()))
#
#
#
#
#
