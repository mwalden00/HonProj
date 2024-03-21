import torch
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import copy
from copulagp.utils import Plot_Fit
from copulagp import vine as v


device = "cpu" if not torch.cuda.is_available() else "cuda:1"

if __name__ == "__main__":
    with torch.device(device):

        with open("../models/results/lorenz_res.pkl", "rb") as f:
            lorenz_results = pkl.load(f)

        with open("../models/results/rand_lorenz_res.pkl", "rb") as f:
            rand_results = pkl.load(f)

        with open("../data/lorenz_vine_data_0.pkl", "rb") as f:
            data = pkl.load(f)

        try:
            Plot_Fit(
                lorenz_results["models"][0][1].model_init(device),
                data["X"],
                data["Y"],
                "Trajectory 1",
                "Trajectory 3",
                device,
            )
        except TypeError:
            pass
        plt.suptitle(
            "Low Level Copula: Parameterized in Time $x$",
            y=1.08,
        )
        plt.savefig("./low_level_lorenz_copula.png")

        from copulagp.utils import Plot_Fit

        try:
            Plot_Fit(
                rand_results["models"][0][1].model_init(device),
                np.random.random(*data["X"].shape),
                data["Y"],
                "Trajectory 1",
                "Trajectory 3",
                device,
            )
        except TypeError:
            pass
        plt.suptitle(
            "Low Level Copula: Parameterized in Random Values (i.e. Unparameterized)",
            y=1.08,
        )
        plt.savefig("./low_level_random_lorenz_copula.png")

        pupil_model_data = copy.deepcopy(lorenz_results["models"])

        for i, layer in enumerate(pupil_model_data):
            for j, cop_data in enumerate(layer):
                cop = cop_data.model_init(device).marginalize(
                    torch.arange(0, 1, 1.0 / 1500)
                )
                pupil_model_data[i][j] = cop
        pupil_vine = v.CVine(pupil_model_data, torch.arange(0, 1, 1.0 / 1500), device=device)

        random_model_data = copy.deepcopy(rand_results["models"])

        for i, layer in enumerate(random_model_data):
            for j, cop_data in enumerate(layer):
                cop = cop_data.model_init(device).marginalize(torch.rand(1500))
                random_model_data[i][j] = cop
        random_vine = v.CVine(random_model_data, torch.rand(1500), device=device)
        
        print('Extracting entropies...')
        H = random_vine.entropy(v=True)
        print("Unparameterized Entropies:", H)
        H_X = pupil_vine.entropy(v=True)
        print("Parameterized Entropies:", H_X)

        I_Y_X = H.mean() - H_X.mean()
        print("Mutual Info Est. =", I_Y_X)
