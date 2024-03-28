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
    return args


def get_R2(cop, Y):

    buckets = torch.arange(Y.shape[0]).chunk(20)
    Y0_sample = cop.sample()[:, 0]

    def ecdf(i):
        """Empirical CDF in bucket i."""
        vals = []
        for y2 in Y[1][buckets[i]]:
            vals.append(
                len(Y[0][buckets[i]][Y[1][buckets[i]] < y2]) / (len(Y[0][buckets[i]]))
            )
        return vals

    def eccdf(i):
        """Empirical Copula CDF in bucket i utilizing copula samples."""
        vals = []
        for y2 in Y[1][buckets[i]]:
            vals.append(
                len(Y0_sample[buckets[i]][Y[1][buckets[i]] < y2]) / (len(Y0_sample))
            )
        return vals

    cop_ccdfs = [torch.vstack([torch.Tensor(eccdf(i)) for i in range(20)])]
    ecdfs = torch.vstack([torch.Tensor(ecdf(i)) for i in range(20)])
    R2s = torch.Tensor(
        [
            1
            - (
                (((ecdfs - ccdfs) ** 2) / ((ecdfs - 0.5) ** 2).clamp(0.001, 1)).sum(
                    axis=1
                )
            ).mean()
            for ccdfs in cop_ccdfs
        ]
    )
