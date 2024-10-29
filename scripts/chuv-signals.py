# %% Load libraries
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mne import create_info, find_events
from mne.io import RawArray
from pymatreader import read_mat

import resp_audio_sleep

root = Path(resp_audio_sleep.__file__).parent.parent / "data" / "chuv"

# %% Load files
data = dict()
for file in root.glob("*.mat"):
    mat = read_mat(file)
    array = np.vstack((mat["resp"], mat["trg"]))
    info = create_info(["resp", "trg"], sfreq=1024, ch_types=["misc", "stim"])
    data[file.stem] = RawArray(array, info)

# %% Plot respiration signals on a common scale
n_samples = min(raw.times.size for raw in data.values())
f, ax = plt.subplots(3, 1, layout="constrained", sharex=True, sharey=True)
for k, (file, raw) in enumerate(data.items()):
    ax[k].plot(raw.times[:n_samples], raw.get_data(picks="resp")[0, :n_samples])
    ax[k].set_title(file)
    # add events to the plot
    events = find_events(raw, stim_channel="trg")
    events = events[events[:, 2] != 128]
    for event in events:
        if n_samples < event[0]:
            break
        ax[k].axvline(raw.times[event[0]], color="teal", linestyle="--")
