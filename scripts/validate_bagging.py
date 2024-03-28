import numpy as np
import torch
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import copy
from copulagp.utils import Plot_Fit
from copulagp import vine as v
from copulagp.train import train_vine
from copulagp.bvcopula import MixtureCopula
from copulagp.synthetic_data import get_random_vine
from copulagp.select_copula import bagged_vine
import os
import gc
import argparse


device = "cpu" if not torch.cuda.is_available() else "cuda:0"


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
        default=783953529,
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


if torch.cuda.is_available():
    device_list = [f"cuda:{n}" for n in range(torch.cuda.device_count())]
else:
    device_list = ["cpu"]


if __name__ == "__main__":
    args = parser().parse_args()
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    mp.set_start_method("spawn")
    with torch.device(device):

        with open("../models/results/pupil_traj_5_res_partial.pkl", "rb") as f:
            pupil_results = pkl.load(f)

        with open("../data/pupil_vine_data_partial_0.pkl", "rb") as f:
            data = pkl.load(f)

        dim = args.dim
        max_el = args.max_el

        pupil_vine = get_random_vine(
            dim, torch.Tensor(data["X"][-5000:]), device=device, max_el=max_el
        )
        print("True vine: ", [[cop.copulas for cop in l] for l in pupil_vine.layers])

        clean()
        if args.skip_true_ent == 1:
            ent = np.genfromtxt(f"./true_ent_{dim}.csv", delimiter=",")
        else:
            ent = pupil_vine.entropy(mc_size=8000).detach().cpu().numpy()
            ent.tofile(f"./true_ent_{dim}.csv", sep=",")
        print(f"Entropy extraction: {ent.mean()} +/- {2*np.std(ent)}")

        n_estimators = args.n_estimators
        assert 4000 % n_estimators == 0

        print(f"Getting {n_estimators} copulaGP estimators...")
        X = pupil_vine.sample()
        X_train = X.reshape(n_estimators, int(5000 / n_estimators), dim)

        Y = data["X"][-5000:]
        Y_train = Y.reshape(n_estimators, int(5000 / n_estimators))

        for i in range(args.bagged_start, n_estimators):
            try:
                os.mkdir(f"../models/layers/pupil_vine/segments/seg_{i}/")
                os.mkdir(f"../models/results/pupil_segments/")
                os.mkdir(f"../data/segmented_pupil_copulas/")
            except:
                pass

            # print(f"\nSelecting Trial {i} with trajectory choices {choices}")
            # np.savetxt(f"./segmented_pupil/choices/choice_i.txt", choices)

            X_chosen = X_train[i]
            with open(f"../data/segmented_pupil_copulas/data_{i}_0.pkl", "wb") as f:
                pkl.dump(
                    dict(
                        [
                            ("X", Y_train[i]),
                            ("Y", X_chosen.cpu().numpy()),
                        ]
                    ),
                    f,
                )

            print(f"Training {i}-th Copula-GP Vine")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_vine(
                path_data=lambda x: f"../data/segmented_pupil_copulas/data_{i}_{x}.pkl",
                path_models=lambda x: f"../models/layers/pupil_vine/segments/seg_{i}/layer_{x}.pkl",
                path_final=f"../models/results/pupil_segments/pupil_{i}_res.pkl",
                path_logs=lambda a, b: f"./segmented_pupil/{a}/layer_{b}",
                exp=f"Vine on trial {i} Parametrized in Pupil Area",
                light=True,
                device_list=device_list,
                start=args.bagged_start_layer,
            )

        print("\n\nGetting Bagged Vine...")

        vines2bag = []

        for i in range(n_estimators):
            with open(f"../models/results/pupil_segments/pupil_{i}_res.pkl", "rb") as f:
                vines2bag.append(pkl.load(f)["models"])

        mean_vine = bagged_vine(
            vines_data=vines2bag, X=torch.Tensor(Y).to(device), Y=X, device=device
        )

        clean()
        print("Getting Bagged Vine Entropy...")
        if args.skip_ent_bagged == 1:
            ent_pred = np.genfromtxt(f"./{n_estimators}_pred.csv", delimiter=",")
        else:
            ent_pred = mean_vine.entropy().detach().cpu().numpy()
            ent_pred.tofile(f"./{n_estimators}_pred_{dim}.csv", sep=",")
        print(f"Entropy: {ent_pred.mean()} +/- {np.std(ent_pred)}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        X_train = X
        Y_train = Y
        baseline_data = dict([("Y", X_train.cpu().numpy()), ("X", Y_train)])
        with open("../data/segmented_pupil_copulas/baseline_data_0.pkl", "wb") as f:
            pkl.dump(baseline_data, f)

        print("Getting Baseline")
        train_vine(
            path_data=lambda x: f"../data/segmented_pupil_copulas/baseline_data_{x}.pkl",
            path_models=lambda x: f"../models/layers/pupil_vine/segments/baseline/layer_{x}.pkl",
            path_final=f"../models/results/pupil_segments/baseline_res.pkl",
            path_logs=lambda a, b: f"./segmented_pupil/{a}/layer_{b}",
            exp=f"Baseline Vine on {dim} dim random copula data Parametrized in Pupil Area",
            light=True,
            start=args.skip_baseline,
            device_list=device_list,
        )

        with open("../models/results/pupil_segments/baseline_res.pkl", "rb") as f:
            baseline_results = pkl.load(f)
        baseline_model_data = copy.deepcopy(baseline_results["models"])
        print(baseline_model_data)

        print("Getting Entropies...")
        clean()
        for i, layer in enumerate(baseline_model_data):
            for j, cop_data in enumerate(layer):
                cop = cop_data.model_init(device).marginalize(torch.Tensor(Y))
                baseline_model_data[i][j] = cop
        baseline_vine = v.CVine(baseline_model_data, torch.Tensor(Y), device=device)
        if args.skip_ent_baseline:
            baseline_ent = np.genfromtxt("./baseline_{dim}.csv", delimiter=",")
        else:
            baseline_ent = baseline_vine.entropy().detach().cpu().numpy()
            baseline_ent.tofile(f"./baseline_{dim}.csv", sep=",")

        print(f"Baseline ent: {baseline_ent.mean()} +/- {2*np.std(baseline_ent)}")

        print("================================================================")
        print("================================================================")
        print(
            "True: \t{:.6f} +/- {:.6f}".format(
                ent[-1000:].mean(), 2 * np.std(ent[-1000:])
            )
        )
        print("MAE Baseline: \t{:.6f}".format(np.abs(ent - baseline_ent).mean()))
        print("MAE Pred: \t{:.6f}".format(np.abs(ent - ent_pred).mean()))
