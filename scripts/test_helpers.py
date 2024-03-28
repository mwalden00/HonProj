import torch
import gc
import argparse


def clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parser():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--seed",
        type=int,
        nargs="?",
        default=-1,
        help="Random seed to pass to torch and numpy.",
    )
    args.add_argument(
        "--skip_baseline",
        type=int,
        nargs="?",
        default=0,
        help="Layer to skip to for Baseline",
    )
    args.add_argument(
        "--bagged_start",
        type=int,
        nargs="?",
        default=0,
        help="Which model to start at for Bagging training",
    )
    args.add_argument(
        "--bagged_start_layer",
        type=int,
        nargs="?",
        default=0,
        help="Which layer to start at for first model",
    )
    args.add_argument(
        "--skip_ent_baseline",
        type=int,
        nargs="?",
        default=0,
        help="Skip baseline entropy calc",
    )
    args.add_argument(
        "--skip_ent_bagged",
        type=int,
        nargs="?",
        default=0,
        help="Skip bagged entropy calc",
    )
    args.add_argument(
        "--skip_true_ent", type=int, nargs="?", default=0, help="Skip true ent. calc."
    )
    args.add_argument(
        "--n_estimators",
        type=int,
        nargs="?",
        default=4,
        help="Number of copula-gp estimators",
    )
    args.add_argument(
        "--dim", nargs="?", type=int, default=3, help="Dim of random data"
    )
    args.add_argument(
        "--max_el", nargs="?", type=int, default=5, help="Max num. of copulas mixed"
    )
    args.add_argument(
        "--shuffle_bags",
        nargs="?",
        type=int,
        defalt=1,
        help="Shuffle data during training",
    )
    args.add_argument("--shufle", nargs="?", type=int, default=0)
    args.add_argument("--linear_input", nargs="?", type=int, default=0)
    return args


def num_vine_params(model):
    p = 0
    for layer in model.layers:
        for copula in layer:
            p += 2 * len(copula.copulas)
    return p
