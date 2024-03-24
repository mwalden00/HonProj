import torch
import gc
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
        [l[:, :100] for l in traj_and_pupil_data["trajectories"]], axis=1
    )
    for i, traj in enumerate(Y):
        Y[i] = NormalizeData(traj)
    x = NormalizeData(
        np.concatenate([l[:100] for l in traj_and_pupil_data["pupil area"]])
    )

    Y = Y.reshape(100, 100, 13)
    x = x.reshape(100, 100)

    for i in range(100):
        with open(f"../data/segmented_pupil_copulas/data_{i}_0.pkl", "wb") as f:
            pkl.dump(dict([("Y", Y[i]), ("X", x[i])]), f)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_vine(
            path_data=lambda x: f"../data/segmented_pupil_copulas/data_{i}_{x}.pkl",
            path_models=lambda x: f"../models/layers/pupil_vine/segments/seg_{i}/layer_{x}.pkl",
            path_final=f"../models/results/pupil_segments/pupil_{i}_res.pkl",
            path_logs=lambda a, b: f"./segmented_pupil/{a}/layer_{b}",
            exp=f"Vine on trial {i} 13 trajectories Parametrized in Pupil Area",
            light=False,
            device_list=device_list,
        )
