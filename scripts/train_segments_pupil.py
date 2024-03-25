import torch
import gc
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
from copulagp.train import train_vine

# tell pandas to show all columns when we display a DataFrame
pd.set_option("display.max_columns", None)
if torch.cuda.is_available():
    device_list = [f"cuda:{n}" for n in range(torch.cuda.device_count())]
else:
    device_list = ["cpu"]
if __name__ == "__main__":
    print("Devices:", device_list)
    mp.set_start_method("spawn")

    def NormalizeData(data):
        return ((data - np.min(data)) / (np.max(data) - np.min(data))) * 0.99 + 0.001

    with open("../data/processed/pupil_vine_data_partial_0.pkl", "rb") as f:
        traj_and_pupil_data = pkl.load(f)

    X = traj_and_pupil_data["Y"]
    Y = traj_and_pupil_data["X"]

    Y = Y.reshape(100, 100)
    X = X.reshape(100, 100, 5)

    for i in range(0, 10, 4):
        choices = np.random.choice(13, 6, replace=False)
        try:
            os.mkdir(f"../models/layers/pupil_vine/segments/seg_{i}/")
            os.mkdir(f"../models/results/pupil_segments/")
            os.mkdir(f"../data/segmented_pupil_copulas/")
        except:
            pass

        print(f"\nSelecting Trial {i} with trajectory choices {choices}")
        np.savetxt(f"./segmented_pupil/choices/choice_i.txt", choices)

        X_chosen = np.concatenate(np.stack(X[i : i + 4]))[:, choices]

        with open(f"../data/segmented_pupil_copulas/data_{i}_0.pkl", "wb") as f:
            pkl.dump(dict([("X", np.concatenate(Y[i : i + 4])), ("Y", X_chosen)]), f)

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
