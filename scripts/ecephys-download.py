import os
import pandas as pd
from pathlib import Path
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from neo.core import SpikeTrain
import quantities as pq
from tqdm import tqdm
import numpy as np
import pickle as pkl

# tell pandas to show all columns when we display a DataFrame
pd.set_option("display.max_columns", None)

output_dir = os.path.expanduser("~/ecephys/data")
resources_dir = Path.cwd().parent / "resources"
DOWNLOAD_LFP = False

manifest_path = os.path.join(output_dir, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

session_id = 756029989  # for example
session = cache.get_session_data(session_id)


quality_units = session.units[
    (session.units["snr"] > 3) & (session.units["isi_violations"] < 0.05)
]

quality_unit_ids = quality_units.index.values
drifting_gratings_presentation_ids = np.append(
    session.stimulus_presentations.loc[
        (session.stimulus_presentations["stimulus_name"] == "drifting_gratings")
    ].index.values,
    0,
)

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
stims = sorted_spikes["stimulus_presentation_id"].unique()[1:]

spike_train_trials = []
all_spikes = {unit: np.array([]) for unit in quality_unit_ids}
good_spike_dict = {unit: session.spike_times[unit] for unit in quality_unit_ids}

N_trials = 100

# Store raw pupil area
pupil_data = (
    session.get_screen_gaze_data()["raw_pupil_area"]
    .fillna(method="ffill")
    .rolling(
        window=10,
    )
    .mean()
)
min_loc = np.argmin(np.abs(pupil_data.index.values - 1585.734418))
max_loc = np.argmin(np.abs(pupil_data.index.values - 2185.235561))
pupil_data = pupil_data.iloc[min_loc:max_loc]

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
    # We will use these to get the behavioral variable (pupil dilation)
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
    SpikeTrain(
        all_spikes[unit],
        t_start=spike_train_trials[0][0][0].t_start,
        t_stop=spike_train_trials[-1][0][0].t_stop,
        units=pq.s,
    )
    for unit in quality_unit_ids
]

# Dump Data
times.to_csv("../data/processed/quality_spike_times_drifting_gratings.csv")
with open("../data/processed/spike_train_trials_drifting_gratings.pkl", "wb") as f:
    pkl.dump(spike_train_trials, f)
with open("../data/processed/single_trial_drfting_gratings.pkl", "wb") as f:
    pkl.dump(all_spikes, f)
with open("../data/processed/pupil_data.pkl", "wb") as f:
    pkl.dump(pupil_data, f)
