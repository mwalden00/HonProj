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
import os
import gc


device = "cpu" if not torch.cuda.is_available() else "cuda:0"

pd.set_option("display.max_columns", None)
if torch.cuda.is_available():
    device_list = [f"cuda:{n}" for n in range(torch.cuda.device_count())]
else:
    device_list = ["cpu"]


def get_mean_copula(
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
        cop_data.model_init(device).marginalize(X.to(device)) for cop_data in cop_datas
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

    return MixtureCopula(theta=thetas, mix=mixes, copulas=cop_list, rotations=rotations)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    with torch.device(device):

        with open("../models/results/pupil_traj_13_res.pkl", "rb") as f:
            pupil_results = pkl.load(f)

        with open("../models/results/random_traj_13_res.pkl", "rb") as f:
            rand_results = pkl.load(f)

        with open("../data/pupil_vine_data_0.pkl", "rb") as f:
            data = pkl.load(f)

        pupil_model_data = copy.deepcopy(pupil_results["models"])

        print("Instatiating pupil vine object...\n")

        for i, layer in enumerate(pupil_model_data):
            for j, cop_data in enumerate(layer):
                cop = cop_data.model_init(device).marginalize(torch.Tensor(data["X"]))
                pupil_model_data[i][j] = cop
        pupil_vine = v.CVine(pupil_model_data, torch.Tensor(data["X"]), device=device)

        X = pupil_vine.sample().reshape(100, 100, 13)
        Y = data["X"].reshape(100, 100)

        for i in range(0, 25):
            try:
                os.mkdir(f"../models/layers/pupil_vine/segments/seg_{i}/")
                os.mkdir(f"../models/results/pupil_segments/")
                os.mkdir(f"../data/segmented_pupil_copulas/")
            except:
                pass

            # print(f"\nSelecting Trial {i} with trajectory choices {choices}")
            # np.savetxt(f"./segmented_pupil/choices/choice_i.txt", choices)

            X_chosen = X[i : i + 4].reshape(400, 13)

            with open(f"../data/segmented_pupil_copulas/data_{i}_0.pkl", "wb") as f:
                pkl.dump(
                    dict([("X", np.concatenate(Y[i : i + 4])), ("Y", X_chosen)]), f
                )

            print("Training {i}-th Copula-GP Vine")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_vine(
                path_data=lambda x: f"../data/segmented_pupil_copulas/data_{i}_{x}.pkl",
                path_models=lambda x: f"../models/layers/pupil_vine/segments/seg_{i}/layer_{x}.pkl",
                path_final=f"../models/results/pupil_segments/pupil_{i}_res.pkl",
                path_logs=lambda a, b: f"./segmented_pupil/{a}/layer_{b}",
                exp=f"Vine on trial {i} 13 trajectories Parametrized in Pupil Area",
                light=True,
                device_list=device_list,
            )

        os.remove("../data/segmented_pupil_copulas/*.pkl")

        print("\n\nGetting Mean Vine...")

        mean_copulas = [[[] for j in range(12 - i)] in i in range(12)]

        for i in range(25):
            with open(f"../models/results/pupil_segments/pupil_{i}_res.pkl", "rb") as f:
                models_i = pkl.load(f)["models"]

            for l, layer in enumerate(models_i):
                for n, copula in enumerate(layer):
                    mean_copulas[l][n].append(copula.model_init(device))

        n_estimators = 25

        for l, layer in enumerate(mean_copulas):
            for n, copula_data_list in enumerate(layer):
                mean_copulas[l][n] = get_mean_copula(
                    copula_data_list, 25, X[:1500], device=device
                )

        mean_vine = v.vine(mean_copulas, X[:1500], device=device)
        print("Getting Mean Vine Entropy...")
        print("Entropy: ", mean_vine.entropy())