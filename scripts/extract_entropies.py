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


device = "cpu" if not torch.cuda.is_available() else "cuda:0"

if __name__ == "__main__":
    with torch.device(device):
        torch.set_default_device(device)

        with open("../models/results/pupil_traj_13_res.pkl", "rb") as f:
            pupil_results = pkl.load(f)

        with open("../models/results/random_traj_13_res.pkl", "rb") as f:
            rand_results = pkl.load(f)

        with open("../data/pupil_vine_data_0.pkl", "rb") as f:
            data = pkl.load(f)

        try:
            Plot_Fit(
                pupil_results["models"][0][1].model_init(device),
                data["X"],
                data["Y"],
                "Trajectory 1",
                "Trajectory 3",
                device,
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
                rand_results["models"][0][1].model_init(device),
                torch.rand(*data["X"].shape),
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
        plt.savefig("./low_level_random_copula.png")

        pupil_model_data = copy.deepcopy(pupil_results["models"])

        for i, layer in enumerate(pupil_model_data):
            for j, cop_data in enumerate(layer):
                cop = cop_data.model_init(device).marginalize(torch.Tensor(data["X"]))
                pupil_model_data[i][j] = cop
        pupil_vine = v.CVine(pupil_model_data, torch.Tensor(data["X"]), device=device)

        random_model_data = copy.deepcopy(pupil_results["models"])

        for i, layer in enumerate(random_model_data):
            for j, cop_data in enumerate(layer):
                cop = cop_data.model_init(device).marginalize(torch.rand(15000))
                random_model_data[i][j] = cop
        random_vine = v.CVine(random_model_data, torch.rand(15000), device=device)
        print("Calculating entropies...")
        H = random_vine.entropy(v=True)
        np.savetxt("./rand_vine_entropies.csv", H.cpu().numpy(), delimiter=",")
        print("Unparameterized Entropies:", H)
        H_X = pupil_vine.entropy(v=True)
        np.savetxt("./pupil_vine_entropies.csv", H_X.cpu().numpy(), delimiter=",")
        print("Parameterized Entropies:", H_X)

        I_Y_X = H.mean() - H_X.mean()
        print("Mutual Info Est. =", I_Y_X)
