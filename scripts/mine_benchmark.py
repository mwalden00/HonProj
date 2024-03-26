import torch
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from copulagp.train import train_vine
from copulagp import vine as v
from copulagp.MI import train_MINE
import copy

# tell pandas to show all columns when we display a DataFrame
pd.set_option("display.max_columns", None)
if torch.cuda.is_available():
    device_list = [f"cuda:{n}" for n in range(torch.cuda.device_count())]
else:
    device_list = ["cpu"]
device = device_list[0]

if __name__ == "__main__":
    mp.set_start_method("spawn")

    def NormalizeData(data):
        return ((data - np.min(data)) / (np.max(data) - np.min(data))) * 0.99 + 0.001

    with open("../data/processed/traj_and_pupil_data.pkl", "rb") as f:
        traj_and_pupil_data = pkl.load(f)

    Y = np.concatenate(
        [l[:5, :100] for l in traj_and_pupil_data["trajectories"]], axis=1
    )
    for i, traj in enumerate(Y):
        Y[i] = NormalizeData(traj)
    x = NormalizeData(
        np.concatenate([l[:100] for l in traj_and_pupil_data["pupil area"]])
    )

    data = dict([("Y", Y.T), ("X", x)])

    print("Getting MINE est...")
    print("MINE MI est: ", train_MINE(y=data["Y"], x=data["X"], H=200))
