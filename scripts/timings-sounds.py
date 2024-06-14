# %% Load libraries
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mne import Epochs, create_info, find_events
from mne.io import RawArray, read_raw_fif
from pyxdf import load_xdf
from scipy.signal import find_peaks

import resp_audio_sleep

# %% Define file
root = Path(resp_audio_sleep.__file__).parent.parent / "data"
fname = "resp-sound-synchronous-1024.fif"

# %% Load XDF file
stream = load_xdf(root / fname)[0][0]
ch_names = [
    ch["label"][0] for ch in stream["info"]["desc"][0]["channels"][0]["channel"]
]
info = create_info(ch_names, 1024, "eeg")
raw = RawArray(stream["time_series"].T, info)
raw.pick(("TRIGGER", "AUX3", "AUX7", "AUX8"))
raw.set_channel_types(
    {"TRIGGER": "stim", "AUX3": "misc", "AUX7": "misc", "AUX8": "misc"}
)

# %% Load FIF file
raw = read_raw_fif(root / fname, preload=True)

# %% Create event-locked epochs
events = find_events(raw, stim_channel="TRIGGER")
epochs = Epochs(
    raw, events, tmin=-0.1, tmax=0.3, baseline=None, picks="AUX3", preload=True
)
epochs.plot(picks="AUX3", n_epochs=1, scalings="auto")

# %% Measure peak to trigger delay
raw_stream = raw.copy()
raw_stream.notch_filter(50, method="iir", phase="forward", picks="misc")
raw_stream.notch_filter(100, method="iir", phase="forward", picks="misc")
raw_stream.crop(5, None)
events = find_events(raw_stream, stim_channel="TRIGGER")
resp_events = events[np.where(events[:, 2] == 1)[0]]
resp_events[:, 0] = resp_events[:, 0] - raw_stream.first_samp
resp = raw_stream.get_data(picks="AUX7").squeeze()
peaks_resp = find_peaks(resp, distance=0.8 * raw.info["sfreq"])[0]


# %% Match closest peak and event
def match_positions(x, y, threshold: int):
    x = np.array(x)
    y = np.array(y)
    d = np.repeat(x, y.shape[0]).reshape(x.shape[0], y.shape[0])
    d -= y
    idx, idy = np.where((-threshold < d) & (d < threshold))
    assert idx.shape == idy.shape  # sanity-check
    return idx, idy


idx_resp, idx_resp_events = match_positions(
    peaks_resp, resp_events[:, 0], 0.5 * raw.info["sfreq"]
)
peaks_resp = peaks_resp[idx_resp]
resp_events = resp_events[idx_resp_events, 0]

# %% Compute and plot delays
delays_resp = resp_events - peaks_resp
f, ax = plt.subplots(1, 1, layout="constrained")
ax.set_title(f"Respiration channel (n={len(delays_resp)})")
ax.hist(delays_resp, bins=40)
