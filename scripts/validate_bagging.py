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
import os
import gc


device = "cpu" if not torch.cuda.is_available() else "cuda:0"

pd.set_option("display.max_columns", None)
if torch.cuda.is_available():
    device_list = [f"cuda:{n}" for n in range(torch.cuda.device_count())]
else:
    device_list = ["cpu"]

if __name__ == "__main__":
    with torch.device(device):

        with open("../models/results/pupil_traj_13_res.pkl", "rb") as f:
            pupil_results = pkl.load(f)

        with open("../models/results/random_traj_13_res.pkl", "rb") as f:
            rand_results = pkl.load(f)

        with open("../data/pupil_vine_data_0.pkl", "rb") as f:
            data = pkl.load(f)

        pupil_model_data = copy.deepcopy(pupil_results["models"])

        print("Instatiating pupil vine object...")

        for i, layer in enumerate(pupil_model_data):
            for j, cop_data in enumerate(layer):
                cop = cop_data.model_init(device).marginalize(torch.Tensor(data["X"]))
                pupil_model_data[i][j] = cop
        pupil_vine = v.CVine(pupil_model_data, torch.Tensor(data["X"]), device=device)

        X = pupil_vine.sample().reshape(100, 100, 13)
        Y = data["X"].reshape(100, 100)

        for i in range(0, 100, 4):
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
