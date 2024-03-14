# first we need a bit of import boilerplate
import os
import pickle as pkl
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from neo import SpikeTrain
import quantities as pq
from elephant.gpfa import GPFA

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import pickle as pkl

# tell pandas to show all columns when we display a DataFrame
pd.set_option("display.max_columns", None)

output_dir = os.path.expanduser("~/ecephys/data")
resources_dir = Path.cwd().parent / "resources"
DOWNLOAD_LFP = False

# Example cache directory path, it determines where downloaded data will be stored
manifest_path = os.path.join(output_dir, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
session_id = 756029989  # for example
session = cache.get_session_data(session_id)

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
quality_units = session.units[
    (session.units["snr"] > 4) & (session.units["isi_violations"] < 0.2)
]

quality_unit_ids = quality_units.index.values
spike_train_trials = []
all_spikes = {unit: np.array([]) for unit in quality_unit_ids}
good_spike_dict = {unit: session.spike_times[unit] for unit in quality_unit_ids}

N_trials = 100

times = session.presentationwise_spike_times(
    unit_ids=quality_unit_ids,
    stimulus_presentation_ids=session.stimulus_presentations.index.values,
)
min_loc = np.argmin(np.abs(times.index.values - 1585.734418))
max_loc = np.argmin(np.abs(times.index.values - 2185.235561))
times = times.iloc[min_loc:max_loc]

# sort the spikes in stimulus_presentation_id, units, and time_since_stimulus_onset.
# In other words, we sort individual presentation data by unit chronologically
sorted_spikes = times.sort_values(
    by=["stimulus_presentation_id", "unit_id", "time_since_stimulus_presentation_onset"]
)

# get all stimulus presentation ids
stims = sorted_spikes["stimulus_presentation_id"].unique()

# get spike trains for the given ids
for stim in tqdm(
    stims[:N_trials],
    desc=f"Getting spike trains for {N_trials} drifting_grating stimulus trials",
):

    # Get the start and stop time for the spike train
    t_start = session.get_stimulus_table().loc[stim]["start_time"]
    t_stop = session.get_stimulus_table().loc[stim + 1]["start_time"]

    # print(stim, t_start, t_stop)

    # Get the spike buckets
    # We will use these to get the running speeds
    buckets = np.arange(np.round(t_start, 2), np.round(t_stop, 2), 0.02)
    if buckets.shape[0] == 151:
        buckets = buckets[1:]
    try:
        assert buckets.shape[0] == 150
    except AssertionError:
        print(f"Time buckets for stimulus is too large / small! Stim: {stim}")

    spike_trains = []

    for unit in quality_unit_ids:
        good_spikes_for_unit = good_spike_dict[unit]
        first_spike = np.argmin(np.abs(good_spikes_for_unit - t_start))
        last_spike = np.argmin(np.abs(good_spikes_for_unit - t_stop))
        while good_spikes_for_unit[first_spike] < t_start:
            first_spike = first_spike + 1
        while good_spikes_for_unit[last_spike] > t_stop:
            last_spike = last_spike - 1
        spike_trains.append(
            SpikeTrain(
                good_spikes_for_unit[first_spike:last_spike],
                t_start=t_start,
                t_stop=t_stop,
                units=pq.s,
            )
        )
        all_spikes[unit] = np.concatenate(
            [all_spikes[unit], good_spikes_for_unit[first_spike:last_spike]]
        )

    # Append trial spike trains to the full list of trials
    spike_train_trials.append((spike_trains, buckets))

all_spikes = [
    [
        SpikeTrain(
            all_spikes[unit][all_spikes[unit] < spike_train_trials[49][0][0].t_stop],
            t_start=spike_train_trials[0][0][0].t_start,
            t_stop=spike_train_trials[49][0][0].t_stop,
            units=pq.s,
        )
        for unit in quality_unit_ids
    ],
    [
        SpikeTrain(
            all_spikes[unit][
                (all_spikes[unit] < spike_train_trials[-1][0][0].t_stop)
                & (all_spikes[unit] > spike_train_trials[50][0][0].t_start)
            ],
            t_start=spike_train_trials[50][0][0].t_start,
            t_stop=spike_train_trials[-1][0][0].t_stop,
            units=pq.s,
        )
        for unit in quality_unit_ids
    ],
]

gpfa_best_dim = GPFA(bin_size=20 * pq.ms, x_dim=15)
gpfa_single_trial = gpfa_best_dim.fit_transform(spiketrains=[all_spikes])

with open("../data/gpfa_data_long.pkl", "wb") as f:
    pkl.dump(gpfa_single_trial, f)
