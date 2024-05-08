# %% Load libraries
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mne import create_info, find_events
from mne.io import RawArray, read_raw_fif
from pyxdf import load_xdf
from scipy.signal import find_peaks

import resp_audio_sleep

# %% Define file
root = Path(resp_audio_sleep.__file__).parent.parent / "data"
fname = "resp-ecg-1024.fif"

# %% Load XDF file
stream = load_xdf(root / fname)[0][0]
ch_names = [
    ch["label"][0] for ch in stream["info"]["desc"][0]["channels"][0]["channel"]
]
info = create_info(ch_names, 1024, "eeg")
raw = RawArray(stream["time_series"].T, info)
raw.pick(("TRIGGER", "AUX7", "AUX8"))
raw.set_channel_types({"TRIGGER": "stim", "AUX7": "misc", "AUX8": "misc"})

# %% Load FIF file
raw = read_raw_fif(root / fname, preload=True)

# %% Create copies with IIR forward and FIR zero-phase filters
raw_stream = raw.copy()
raw_stream.notch_filter(50, method="iir", phase="forward", picks="misc")
raw_stream.notch_filter(100, method="iir", phase="forward", picks="misc")
raw_filter = raw.copy()
raw_filter.filter(None, 20, picks="misc")
raw.crop(9, None)
raw_stream.crop(9, None)
raw_filter.crop(9, None)

# %% Compare raw signal, stream/detector signal and FIR filter signal
f, ax = plt.subplots(2, 1, sharex=True, layout="constrained")
ax[0].set_title("ECG channel")
ax[0].plot(raw.times, raw.get_data(picks="AUX8").squeeze(), color="red", label="Raw")
ax[0].plot(
    raw_stream.times, raw_stream.get_data(picks="AUX8").squeeze(), label="IIR forward"
)
ax[0].plot(
    raw_filter.times,
    raw_filter.get_data(picks="AUX8").squeeze(),
    color="yellow",
    linestyle="--",
    label="FIR zero-phase",
)
ax[1].set_title("Respiration channel")
ax[1].plot(raw.times, raw.get_data(picks="AUX7").squeeze(), color="red")
ax[1].plot(raw_stream.times, raw_stream.get_data(picks="AUX7").squeeze())
ax[1].plot(
    raw_filter.times,
    raw_filter.get_data(picks="AUX7").squeeze(),
    color="yellow",
    linestyle="--",
)
f.legend(loc="outside right")

# %% Find peaks and events on both channels
events = find_events(raw_stream, stim_channel="TRIGGER")
ecg_events = events[np.where(events[:, 2] == 1)[0]]
ecg_events[:, 0] = ecg_events[:, 0] - raw_stream.first_samp
resp_events = events[np.where(events[:, 2] == 2)[0]]
resp_events[:, 0] = resp_events[:, 0] - raw_stream.first_samp
ecg = raw_stream.get_data(picks="AUX8").squeeze()
peaks_ecg = find_peaks(
    ecg, height=np.percentile(ecg, 98), distance=0.2 * raw.info["sfreq"]
)[0]
resp = raw_stream.get_data(picks="AUX7").squeeze()
peaks_resp = find_peaks(resp, prominence=20, distance=0.8 * raw.info["sfreq"])[0]

f, ax = plt.subplots(2, 1, sharex=True, layout="constrained")
ax[0].set_title("ECG channel")
ax[0].plot(raw_stream.times, ecg)
for peak in peaks_ecg:
    ax[0].axvline(raw_stream.times[peak], color="green", linestyle="--")
for event in ecg_events:
    ax[0].axvline(raw_stream.times[event[0]], color="red", linestyle="--")
ax[1].set_title("Respiration channel")
ax[1].plot(raw_stream.times, resp)
for peak in peaks_resp:
    ax[1].axvline(raw_stream.times[peak], color="green", linestyle="--")
for event in resp_events:
    ax[1].axvline(raw_stream.times[event[0]], color="red", linestyle="--")


# %% Match closest peak and event
def match_positions(x, y, threshold: int):
    x = np.array(x)
    y = np.array(y)
    d = np.repeat(x, y.shape[0]).reshape(x.shape[0], y.shape[0])
    d -= y
    idx, idy = np.where((-threshold < d) & (d < threshold))
    assert idx.shape == idy.shape  # sanity-check
    return idx, idy


idx_ecg, idx_ecg_events = match_positions(
    peaks_ecg, ecg_events[:, 0], 0.1 * raw.info["sfreq"]
)
idx_resp, idx_resp_events = match_positions(
    peaks_resp, resp_events[:, 0], 0.5 * raw.info["sfreq"]
)
peaks_ecg = peaks_ecg[idx_ecg]
ecg_events = ecg_events[idx_ecg_events, 0]
peaks_resp = peaks_resp[idx_resp]
resp_events = resp_events[idx_resp_events, 0]

# %% Compute and plot delays
delays_ecg = ecg_events - peaks_ecg
delays_resp = resp_events - peaks_resp
f, ax = plt.subplots(1, 2, layout="constrained")
ax[0].set_title(f"ECG channel (n={len(delays_ecg)})")
ax[0].hist(delays_ecg, bins=20)
ax[1].set_title(f"Respiration channel (n={len(delays_resp)})")
ax[1].hist(delays_resp, bins=20)
