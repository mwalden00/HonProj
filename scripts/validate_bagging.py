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
from copulagp.bvcopula import MixtureCopula, IndependenceCopula
import os
import gc
import argparse


device = "cpu" if not torch.cuda.is_available() else "cuda:0"


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
    return args


if torch.cuda.is_available():
    device_list = [f"cuda:{n}" for n in range(torch.cuda.device_count())]
else:
    device_list = ["cpu"]


def bagged_copula(
    copula_data_list: list,
    n_estimators: int,
    X: torch.Tensor,
    device: torch.device = torch.device("cpu"),
):
    """
    Estimate the copula via pre-trained copula-GP object.
    Calculates average mixed copula via averaging mixing
    and theta params (copula variant wise)
    -----------------
    copula_data_list : List( CopulaData )
        List of Copula-GP Data objects.
        Marginalizing along X gives a distribution over X.
    X : torch.Tensor on range [0,1]
        Tensor of marginalizing variable.
    device : torch.device
        Device ot marginalize on.
    """
    cop_datas = copula_data_list
    assert len(cop_datas) == n_estimators
    N = 0
    cop_indeces = dict()
    cop_combo_indeces = dict()
    cop_combinations = set()
    cop_counts = dict()
    rotations = []

    # Get Rotation and Index Information
    for i, cop_data in enumerate(cop_datas):
        for n, cop in enumerate(cop_data.bvcopulas):
            cop_combo = (cop[0], cop[1])
            if cop_combo not in cop_combinations:
                cop_combinations.add(cop_combo)
                cop_combo_indeces[cop_combo] = N
                cop_counts[N] = 0.0
                N = N + 1
                rotations.append(cop_combo[1])
            idx = cop_combo_indeces[cop_combo]
            cop_counts[idx] = cop_counts[idx] + 1.0
            cop_indeces[(i, n)] = idx

    # Marginalize
    cops = [
        cop_data.model_init(device).marginalize(torch.Tensor(X).to(device))
        for cop_data in cop_datas
    ]

    # Create Mixture as Average
    cop_list = [None for i in range(N)]
    thetas = torch.zeros((N, X.shape[0]))
    mixes = torch.zeros((N, X.shape[0]))

    for i, cop in enumerate(cops):
        for n, cop_type in enumerate(cop.copulas):
            idx = cop_indeces[(i, n)]
            cop_list[idx] = cop_type
            thetas[idx] = thetas[idx] + cop.theta[n] / cop_counts[idx]
            mixes[idx] = mixes[idx] + cop.mix[n] / len(cop_datas)

    print(cop_list)

    if N == 1:
        return (
            cop_list[0](theta=torch.tensor([], device=device))
            if cop_list[0] is IndependenceCopula
            else cop_list[0](theta=thetas, rotation=rotations[0])
        )
    return MixtureCopula(theta=thetas, mix=mixes, copulas=cop_list, rotations=rotations)


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

        pupil_model_data = copy.deepcopy(pupil_results["models"])

        print("Instatiating initial pupil vine object...\n")

        for i, layer in enumerate(pupil_model_data):
            for j, cop_data in enumerate(layer):
                cop = cop_data.model_init(device).marginalize(
                    torch.Tensor(data["X"][-5000:])
                )
                pupil_model_data[i][j] = cop
        pupil_vine = v.CVine(
            pupil_model_data, torch.Tensor(data["X"][-5000:]), device=device
        )

        if args.skip_true_ent == 1:
            ent = np.genfromtxt("./true_ent.csv", delimiter=",")
        else:
            ent = pupil_vine.entropy().detach().cpu().numpy()
            ent.tofile("./true_ent.csv", sep=",")
        print(f"Entropy extraction: {ent.mean()} +/- {2*np.std(ent)}")

        print("Getting vines...")
        X = pupil_vine.sample()
        X_train = X[:4000].reshape(10, 400, 5)

        Y = data["X"][-5000:]
        Y_train = Y[:4000].reshape(10, 400)
        Y_test = Y[-1000:]

        for i in range(args.bagged_start, 10):
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

        bagged_copulas = [[[] for j in range(4 - i)] for i in range(4)]

        for i in range(10):
            with open(f"../models/results/pupil_segments/pupil_{i}_res.pkl", "rb") as f:
                models_i = pkl.load(f)["models"]

            for l, layer in enumerate(models_i):
                for n, copula in enumerate(layer):
                    bagged_copulas[l][n].append(models_i[l][n])

        n_estimators = 10

        for l, layer in enumerate(bagged_copulas):
            for n, copula_data_list in enumerate(layer):
                bagged_copulas[l][n] = bagged_copula(
                    copula_data_list, n_estimators, Y_test, device=device
                )

        mean_vine = v.vine(bagged_copulas, Y_test, device=device)
        print("Getting Bagged Vine Entropy...")
        if args.skip_ent_bagged == 1:
            ent_pred = np.genfromtxt("./pred.csv")
        else:
            ent_pred = mean_vine.entropy().detach().cpu().numpy()
            ent_pred.tofile("./pred.csv", sep=",")
        print(f"Entropy: {ent_pred.mean()} +/- {np.std(ent_pred)}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        X_train = X_train.reshape(4000, 5)
        Y_train = Y_train.reshape(4000)
        baseline_data = dict([("Y", X_train), ("X", Y_train)])
        with open("../data/segmented_pupil_copulas/baseline_data_0.pkl", "wb") as f:
            pkl.dump(baseline_data, f)

        print("Getting Baseline")
        train_vine(
            path_data=lambda x: f"../data/segmented_pupil_copulas/baseline_data_{x}.pkl",
            path_models=lambda x: f"../models/layers/pupil_vine/segments/baseline/layer_{x}.pkl",
            path_final=f"../models/results/pupil_segments/baseline_res.pkl",
            path_logs=lambda a, b: f"./segmented_pupil/{a}/layer_{b}",
            exp=f"Baseline Vine on 5 of 13 trajectories Parametrized in Pupil Area",
            light=True,
            start=args.baseline_skip,
            device_list=device_list,
        )

        with open("../models/results/pupil_segments/baseline_res.pkl", "rb") as f:
            baseline_results = pkl.load(f)
        baseline_model_data = copy.deepcopy(pupil_results["models"])

        print("Getting Entropies...")

        for i, layer in enumerate(pupil_model_data):
            for j, cop_data in enumerate(layer):
                cop = cop_data.model_init(device).marginalize(Y_test)
                baseline_model_data[i][j] = cop
        baseline_vine = v.CVine(
            baseline_model_data, torch.Tensor(Y_test), device=device
        )
        if args.skip_ent_baseline:
            baseline_ent = np.genfromtxt("./baseline.csv", delimiter=",")
        else:
            baseline_ent = baseline_vine.entropy()
            baseline_ent.tofile("./baseline.csv", sep=",")

        print(f"Baseline ent: {baseline_ent} +/- {2*np.std(baseline_ent)}")

        print("================================================================")
        print("================================================================")
        print("True: \t{:.6f} +/- {:.6f}".format(ent, np.std(ent) * 2))
        print("MAE Baseline: \t{:.6f}".format(np.abs(ent - baseline_ent)))
        print("MAE Pred: \t{:.6f}".format(np.abs(ent - ent_pred)))
