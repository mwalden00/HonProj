import pickle as pkl
import numpy as np
from elephant.gpfa import GPFA
import quantities as pq
import pandas as pd
import copy
from sklearn.preprocessing import robust_scale

with open("../data/processed/spike_train_trials_drifting_gratings.pkl", "rb") as f:
    spike_train_trials = pkl.load(f)
with open("../data/processed/pupil_data.pkl", "rb") as f:
    pupil_data = pkl.load(f)

gpfa = GPFA(bin_size=20 * pq.ms, x_dim=13)

traj_data = gpfa.fit_transform(spiketrains=[trial for (trial, _) in spike_train_trials])

bucket_lists = [buckets for (_, buckets) in spike_train_trials]


def get_nearest_val(s1, s2):
    min_idxs = [np.argmin(np.abs(s2.index.values - val)) for val in s1]
    return s2.iloc[min_idxs]


pupil_lists = copy.deepcopy(bucket_lists)
for i, buckets in enumerate(bucket_lists):
    pupil_lists[i] = get_nearest_val(buckets, pupil_data)

pupil_lists = robust_scale(pupil_lists, axis=1)

traj_pupil_data = {"trajectories": traj_data, "pupil area": pupil_lists}
with open("../data/processed/traj_and_pupil_data.pkl", "wb") as f:
    pkl.dump(traj_pupil_data, f)
