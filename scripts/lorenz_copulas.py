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

with torch.device(device=device_list[0]):
    if __name__ == "__main__":
        mp.set_start_method("spawn")

        train_vine(
            path_data=lambda x: f"../data/lorenz_vine_data_{x}.pkl",
            path_models=lambda x: f"../models/layers/lorenz_vine/layer_{x}.pkl",
            path_final=f"../models/results/lorenz_res.pkl",
            path_logs=lambda a, b: f"./{a}/layer_{b}",
            exp=f"Vine on lorenz trajectories parametrized in time",
            start=0,
            device_list=device_list,
        )

        with open("../data/lorenz_vine_data_0.pkl", "rb") as f:
            data = pkl.load(f)
            x = data["X"]
            Y = data["Y"]
        x_rand = np.random.random(x.shape)

        with open("../data/random_lorenz_data_0.pkl", "wb") as f:
            pkl.dump(dict([("Y", Y), ("X", x_rand)]), f)

        train_vine(
            path_data=lambda x: f"../data/random_lorenz_data_{x}.pkl",
            path_models=lambda x: f"../models/layers/random_lorenz/layer_{x}.pkl",
            path_final=f"../models/results/random_lorenz_res.pkl",
            path_logs=lambda a, b: f"./{a}/layer_{b}",
            exp=f"Vine on lorenz trajectories Parametrized in Random Values",
            start=0,
            device_list=device_list,
        )
