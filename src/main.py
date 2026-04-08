import argparse
from src.run import Run
from src.logging_util import set_up_logging
from itertools import product

from dataclasses import asdict


from src.config_parser import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--terminal", choices=["Name", "Deitic", "Base"])
    parser.add_argument("--strategy", choices=["Restricted", "Clone", "Free"])

    args = parser.parse_args()
    config = (
        load_config(args.config) if args.config else load_config("configs/default.toml")
    )
    sensing_terminals = [args.terminal] if args.terminal else config.terminals
    breeding_strategies = [args.strategy] if args.strategy else config.strategies

    logger = set_up_logging(config.title)
    logger.info(asdict(config))

    for terminal, strat in product(sensing_terminals, breeding_strategies):
        run = Run(terminal, strat, logger, config)
        run.run_gens()
        run.save_positions()
        run.save_run_results()
        run.save_pride_history()
