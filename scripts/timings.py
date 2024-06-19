# %% Load libraries
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mne import Epochs, find_events
from mne.io import read_raw_fif
from scipy.signal import find_peaks

import resp_audio_sleep

root = Path(resp_audio_sleep.__file__).parent.parent / "data"


# %% Isochronous
fname = root / "isochronous-raw.fif"
raw = read_raw_fif(fname, preload=True)
events = find_events(raw)
events = events[events[:, 2] == 3]  # only keep the target events
epochs = Epochs(raw, events, tmin=-0.05, tmax=0.25, picks="misc")
epochs.plot(picks="AUX9", n_epochs=1, events=events, scalings=dict(misc=4e5))

# the target delay was 0.8 seconds
delays_sample = np.diff(events[:, 0])
delays_ms = delays_sample * 1000 / raw.info["sfreq"]
f, ax = plt.subplots(1, 2, layout="constrained")
f.suptitle("Task: isochronous - target: 800 ms")
ax[0].hist(
    delays_sample,
    bins=np.arange(np.min(delays_sample) - 0.5, np.max(delays_sample) + 1.5, 1),
    edgecolor="black",
)
ax[0].set_title("Distribution of delays in samples")
ax[0].set_xlabel("Samples")
ax[0].set_xticks(np.arange(np.min(delays_sample), np.max(delays_sample) + 1, 1))
ax[1].hist(
    delays_ms,
    bins=np.arange(
        np.min(delays_ms) - 0.5 * 1000 / raw.info["sfreq"],
        np.max(delays_ms) + 1.5 * 1000 / raw.info["sfreq"],
        1000 / raw.info["sfreq"],
    ),
    edgecolor="black",
)
ax[1].set_title("Distribution of delays in ms")
ax[1].set_xlabel("ms")
ax[1].set_xticks(
    np.arange(
        np.min(delays_ms),
        np.max(delays_ms) + 1000 / raw.info["sfreq"],
        1000 / raw.info["sfreq"],
    )
)


# %% Asynchronous
fname = root / "asynchronous-raw.fif"
raw = read_raw_fif(fname, preload=True)
events = find_events(raw)
events = events[events[:, 2] == 3]  # only keep the target events
epochs = Epochs(raw, events, tmin=-0.05, tmax=0.25, picks="misc")
epochs.plot(picks="AUX9", n_epochs=1, events=events, scalings=dict(misc=4e5))

# the target delay was 0.8 to 1.2 seconds (uniformly distributed)
delays_sample = np.diff(events[:, 0])
delays_ms = delays_sample * 1000 / raw.info["sfreq"]
f, ax = plt.subplots(1, 2, layout="constrained")
f.suptitle("Task: asynchronous - target: 800-1200 ms")
ax[0].hist(
    delays_sample,
    bins=np.arange(np.min(delays_sample) - 10, np.max(delays_sample) + 11, 20),
    edgecolor="black",
)
ax[0].set_title("Distribution of delays in samples")
ax[0].set_xlabel("Samples")
ax[0].set_xticks(np.arange(np.min(delays_sample), np.max(delays_sample) + 1, 100))
ax[1].hist(
    delays_ms,
    bins=np.arange(
        np.min(delays_ms) - 10 * 1000 / raw.info["sfreq"],
        np.max(delays_ms) + 11 * 1000 / raw.info["sfreq"],
        20 * 1000 / raw.info["sfreq"],
    ),
    edgecolor="black",
)
ax[1].set_title("Distribution of delays in ms")
ax[1].set_xlabel("ms")
ax[1].set_xticks(
    np.arange(
        np.min(delays_ms),
        np.max(delays_ms) + 1000 / raw.info["sfreq"],
        100 * 1000 / raw.info["sfreq"],
    )
)


# %% Synchronous respiration
fname = root / "synchronous-respiration-raw.fif"
raw = read_raw_fif(fname, preload=True)
events = find_events(raw)
events = events[events[:, 2] == 3]  # only keep the target events
epochs = Epochs(raw, events, tmin=-0.05, tmax=0.25, picks="misc")
epochs.plot(picks="AUX9", n_epochs=1, events=events, scalings=dict(misc=4e5))

# filter the signal with the same filters as the detector
raw.notch_filter(50, picks="AUX7", method="iir", phase="forward")
raw.notch_filter(100, picks="AUX7", method="iir", phase="forward")
raw.filter(None, 20, picks="AUX7", method="iir", phase="forward")
raw.crop(5, None)  # time for the filter without initial state to settle
data = raw.get_data(picks="AUX7").squeeze()
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


# %% Synchronous cardiac
fname = root / "synchronous-cardiac-raw.fif"
raw = read_raw_fif(fname, preload=True)
events = find_events(raw)
events = events[events[:, 2] == 3]  # only keep the target events
epochs = Epochs(raw, events, tmin=-0.05, tmax=0.25, picks="misc")
epochs.plot(picks="AUX9", n_epochs=1, events=events, scalings=dict(misc=4e5))

# filter the signal with the same filters as the detector
raw.notch_filter(50, picks="AUX8", method="iir", phase="forward")
raw.notch_filter(100, picks="AUX8", method="iir", phase="forward")
raw.crop(5, None)  # time for the filter without initial state to settle
data = raw.get_data(picks="AUX8").squeeze()

# the height constrain is badly apply offline (the rolling window might be different),
# thus we need to confirm visually that it looks good.
f, ax = plt.subplots(1, 1, layout="constrained")
cardiac_peaks = list()
winsize = int(10 * raw.info["sfreq"])  # 10 seconds
for k in range(0, data.size, winsize):
    data_ = np.copy(data[k : k + winsize])
    times_ = np.copy(raw.times[k : k + winsize])
    z = np.polyfit(times_, data_, 1)
    data_ -= z[0] * times_ + z[1]
    height = np.percentile(data_, 98.5)
    peaks = find_peaks(data_, distance=0.3 * raw.info["sfreq"], height=height)[0]
    cardiac_peaks.extend(peaks + k)
    # iterative plot
    ax.plot(times_, data_, color="blue")
    ax.axhline(y=height, color="red", linestyle="--", xmin=times_[0], xmax=times_[-1])
    for peak in peaks:
        ax.axvline(times_[peak], color="red", linestyle="--")
cardiac_peaks = np.array(cardiac_peaks)

events = find_events(raw)
events = events[events[:, 2] == 3]  # only keep the target events
events[:, 0] -= raw.first_samp
for event in events:
    ax.axvline(raw.times[event[0]], color="black", linestyle="--")

# target was 1.8 to 2.2 seconds
cardiac_delays = np.diff(cardiac_peaks)
event_delays = np.diff(events[:, 0])
f, ax = plt.subplots(1, 2, layout="constrained")
f.suptitle("Task: synchronous cardiac - target: 1800-2200 ms")
ax[0].hist(cardiac_delays, edgecolor="black")
ax[1].hist(event_delays, edgecolor="black")
ax[0].set_title("Delay between R-peaks")
ax[1].set_title("Delay between triggers (sounds)")
ax[0].set_xlabel("Samples")
ax[1].set_xlabel("Samples")

f, ax = plt.subplots(1, 2, layout="constrained")
f.suptitle("Task: synchronous cardiac - target: 1800-2200 ms")
ax[0].hist(cardiac_delays / raw.info["sfreq"], edgecolor="black")
ax[1].hist(event_delays / raw.info["sfreq"], edgecolor="black")
ax[0].set_title("Delay between R-peaks")
ax[1].set_title("Delay between triggers (sounds)")
ax[0].set_xlabel("ms")
ax[1].set_xlabel("ms")


# r-peak to event delay
def match_positions(x, y, threshold: int):
    x = np.array(x)
    y = np.array(y)
    d = np.repeat(x, y.shape[0]).reshape(x.shape[0], y.shape[0])
    d -= y
    idx, idy = np.where((-threshold < d) & (d < threshold))
    assert idx.shape == idy.shape  # sanity-check
    return idx, idy


idx_cardiac, idx_events = match_positions(
    cardiac_peaks, events[:, 0], 0.5 * raw.info["sfreq"]
)
cardiac_peaks = cardiac_peaks[idx_cardiac]
events = events[idx_events, 0]
delays_sample = events - cardiac_peaks
delays_ms = delays_sample * 1000 / raw.info["sfreq"]
f, ax = plt.subplots(1, 2, layout="constrained")
f.suptitle("Task: synchronous cardiac - target: 250 ms post R-peak")
ax[0].hist(
    delays_sample,
    bins=np.arange(np.min(delays_sample) - 0.5, np.max(delays_sample) + 1.5, 1),
    edgecolor="black",
)
ax[0].set_title("Distribution of delays in samples")
ax[0].set_xlabel("Samples")
ax[0].set_xticks(np.arange(np.min(delays_sample), np.max(delays_sample) + 1, 1))
ax[1].hist(
    delays_ms,
    bins=np.arange(
        np.min(delays_ms) - 0.5 * 1000 / raw.info["sfreq"],
        np.max(delays_ms) + 1.5 * 1000 / raw.info["sfreq"],
        1000 / raw.info["sfreq"],
    ),
    edgecolor="black",
)
ax[1].set_title("Distribution of delays in ms")
ax[1].set_xlabel("ms")
ax[1].set_xticks(
    np.arange(
        np.min(delays_ms),
        np.max(delays_ms) + 1000 / raw.info["sfreq"],
        1000 / raw.info["sfreq"],
    )
)
