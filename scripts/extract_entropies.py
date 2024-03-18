import torch
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import copy
from copulagp.utils import Plot_Fit
from copulagp import vine as v


device_list = ["cpu" if not torch.cuda.is_available() else "cuda"]
torch.set_default_device(device_list[0])

if __name__ == "__main__":
    with torch.device(device_list[0]):
        torch.set_default_device(device_list[0])

        with open("../models/results/pupil_traj_13_res.pkl", "rb") as f:
            pupil_results = pkl.load(f)

        with open("../models/results/random_traj_13_res.pkl", "rb") as f:
            rand_results = pkl.load(f)

        with open("../data/pupil_vine_data_0.pkl", "rb") as f:
            data = pkl.load(f)

        try:
            Plot_Fit(
                pupil_results["models"][0][1].model_init(device_list[0]),
                data["X"],
                data["Y"],
                "Trajectory 1",
                "Trajectory 3",
                device_list[0],
            )
        except TypeError:
            pass
        plt.suptitle(
            "Low Level Copula: Parameterized in Pupil Dilation Robust Normalized per Trial",
            y=1.08,
        )
        plt.savefig("./low_level_pupil_copula.png")

        from copulagp.utils import Plot_Fit

        try:
            Plot_Fit(
                rand_results["models"][0][1].model_init(device_list[0]),
                np.random.random(*data["X"].shape),
                data["Y"],
                "Trajectory 1",
                "Trajectory 3",
                device_list[0],
            )
        except TypeError:
            pass
        plt.suptitle(
            "Low Level Copula: Parameterized in Random Values (i.e. Unparameterized)",
            y=1.08,
        )
        plt.savefig("./low_level_random_copula.png")

        pupil_model_data = copy.deepcopy(pupil_results["models"])

        for i, layer in enumerate(pupil_model_data):
            for j, cop_data in enumerate(layer):
                cop = cop_data.model_init(device_list[0]).marginalize(
                    torch.arange(0, 1, 0.01)
                )
                pupil_model_data[i][j] = cop
        pupil_vine = v.CVine(pupil_model_data, torch.arange(0, 1, 0.01))

        random_model_data = copy.deepcopy(pupil_results["models"])

        for i, layer in enumerate(random_model_data):
            for j, cop_data in enumerate(layer):
                cop = cop_data.model_init(device_list[0]).marginalize(
                    torch.arange(0, 1, 0.01)
                )
                random_model_data[i][j] = cop
        random_vine = v.CVine(random_model_data, torch.arange(0, 1, 0.01))

        H = random_vine.entropy(v=True)
        H_X = random_vine.entropy(v=True)

        I_Y_X = H.mean() - H_X.mean()
        print("Mutual Info Est. =", I_Y_X)
