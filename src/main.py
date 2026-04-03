import argparse
from node import Run
from logging_util import set_up_logging
from itertools import product

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal", choices=["Name", "Deitic", "Base"])
    parser.add_argument("--strategy", choices=["Restricted", "Clone", "Free"])
    parser.add_argument("--title")

    args = parser.parse_args()

    sensing_terminals = [args.terminal] if args.terminal else ["Name", "Deitic", "Base"]
    breeding_strategies = (
        [args.strategy] if args.strategy else ["Restricted", "Clone", "Free"]
    )
    title = args.title if args.title else ""

    logger = set_up_logging()
    for terminal, strat in product(sensing_terminals, breeding_strategies):
        run = Run(terminal, strat, logger, title)
        run.run_gens()
        run.save_best_positions()
        run.save_run_results()
