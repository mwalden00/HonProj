import torch
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from copulagp.train import train_vine

# tell pandas to show all columns when we display a DataFrame
pd.set_option("display.max_columns", None)
if torch.cuda.is_available():
    device_list = [f"cuda:{n}" for n in range(torch.cuda.device_count())]
else:
    device_list = ["cpu"]

if __name__ == "__main__":
    mp.set_start_method("spawn")

    def NormalizeData(data):
        return ((data - np.min(data)) / (np.max(data) - np.min(data))) * 0.99 + 0.001

    with open("../data/processed/traj_and_pupil_data.pkl", "rb") as f:
        traj_and_pupil_data = pkl.load(f)

    Y = np.concatenate(
        [l[:, :100, :5] for l in traj_and_pupil_data["trajectories"]], axis=1
    )
    for i, traj in enumerate(Y):
        Y[i] = NormalizeData(traj)
    x = NormalizeData(
        np.concatenate([l[:100] for l in traj_and_pupil_data["pupil area"]])
    )

    with open("../data/pupil_vine_data_partial_0.pkl", "wb") as f:
        pkl.dump(dict([("Y", Y.T), ("X", x)]), f)

    N = Y.shape[0]

    train_vine(
        path_data=lambda x: f"../data/pupil_vine_data_partial_{x}.pkl",
        path_models=lambda x: f"../models/layers/pupil_vine/partial_layer_{x}.pkl",
        path_final=f"../models/results/pupil_traj_{N}_res_partial.pkl",
        path_logs=lambda a, b: f"./{a}/layer_{b}",
        exp=f"Vine on {N} of 13 trajectories Parametrized in Pupil Area",
        light=False,
        start=0
        device_list=device_list,
    )
