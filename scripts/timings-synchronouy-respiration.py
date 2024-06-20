# %% Load libraries
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mne import find_events
from mne.io import read_raw_fif
from mne_lsl.stream._filters import create_filter
from scipy.signal import find_peaks, sosfilt

import resp_audio_sleep

root = Path(resp_audio_sleep.__file__).parent.parent / "data"


# %% Synchronous respiration
fname = root / "synchronous-respiration-raw.fif"
raw = read_raw_fif(fname, preload=True)
events = find_events(raw)
events = events[events[:, 2] == 3]  # only keep the target events

# filter the signal with the same filters as the detector
data = raw.get_data(picks="AUX7").T
# 50 Hz notch
notch_widths = 50 / 200
trans_bandwidth = 1
low = 50 - notch_widths / 2.0 - trans_bandwidth / 2.0
high = 50 + notch_widths / 2.0 + trans_bandwidth / 2.0
filter1 = create_filter(raw.info["sfreq"], high, low, iir_params=None)
filter1["zi"] = np.mean(data, axis=0) * filter1["zi_unit"]
notch_widths = 100 / 200
trans_bandwidth = 1
low = 100 - notch_widths / 2.0 - trans_bandwidth / 2.0
high = 100 + notch_widths / 2.0 + trans_bandwidth / 2.0
filter2 = create_filter(raw.info["sfreq"], high, low, iir_params=None)
filter2["zi"] = np.mean(data, axis=0) * filter2["zi_unit"]
filter3 = create_filter(raw.info["sfreq"], None, 20, iir_params=None)
filter3["zi"] = np.mean(data, axis=0) * filter3["zi_unit"]
data, _ = sosfilt(filter1["sos"], data, axis=0, zi=filter1["zi"])
data, _ = sosfilt(filter2["sos"], data, axis=0, zi=filter2["zi"])
data, _ = sosfilt(filter3["sos"], data, axis=0, zi=filter3["zi"])
data = data.squeeze()
z = np.polyfit(raw.times, data, 1)
data -= z[0] * raw.times + z[1]
peaks = find_peaks(data, distance=0.8 * raw.info["sfreq"], height=np.mean(data))[0]

f, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(raw.times, data, color="blue")
for peak in peaks:
    ax.axvline(raw.times[peak], color="red", linestyle="--")
events = find_events(raw)
events = events[events[:, 2] == 3]  # only keep the target events
events[:, 0] -= raw.first_samp
for event in events:
    ax.axvline(raw.times[event[0]], color="black", linestyle="--")


# peak to trigger (sound) delay
def match_positions(x, y, threshold: int):
    x = np.array(x)
    y = np.array(y)
    d = np.repeat(x, y.shape[0]).reshape(x.shape[0], y.shape[0])
    d -= y
    idx, idy = np.where((-threshold < d) & (d < threshold))
    assert idx.shape == idy.shape  # sanity-check
    return idx, idy


idx_resp, idx_events = match_positions(peaks, events[:, 0], 0.8 * raw.info["sfreq"])
resp_peaks = peaks[idx_resp]
events = events[idx_events, 0]
delays_sample = events - resp_peaks
delays_ms = delays_sample * 1000 / raw.info["sfreq"]
f, ax = plt.subplots(1, 2, layout="constrained")
f.suptitle("Task: synchronous respiration - target: 250 ms post R-peak")
ax[0].hist(
    delays_sample,
    bins=np.arange(np.min(delays_sample) - 0.5, np.max(delays_sample) + 1.5, 2),
    edgecolor="black",
)
ax[0].set_title("Distribution of delays in samples")
ax[0].set_xticks(np.arange(np.min(delays_sample), np.max(delays_sample) + 1, 60))
ax[0].set_xlabel("Samples")
ax[1].hist(
    delays_ms,
    bins=np.arange(
        np.min(delays_ms) - 0.5 * 1000 / raw.info["sfreq"],
        np.max(delays_ms) + 1.5 * 1000 / raw.info["sfreq"],
        2000 / raw.info["sfreq"],
    ),
    edgecolor="black",
)
ax[1].set_title("Distribution of delays in ms")
ax[1].set_xticks(
    np.arange(
        np.min(delays_ms),
        np.max(delays_ms) + 1000 / raw.info["sfreq"],
        60000 / raw.info["sfreq"],
    )
)
ax[1].set_xlabel("ms")
