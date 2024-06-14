from pathlib import Path

from matplotlib import pyplot as plt
from mne.io import read_raw_fif
from scipy.signal import find_peaks

import resp_audio_sleep

root = Path(resp_audio_sleep.__file__).parent.parent / "data"
fname = "resp-ecg-1024.fif"

raw = read_raw_fif(root / fname, preload=True)
raw.pick("AUX7")  # respiration channel
raw_stream = raw.copy()
raw_stream.notch_filter(50, method="iir", phase="forward", picks="AUX7")
raw_stream.notch_filter(100, method="iir", phase="forward", picks="AUX7")
raw_stream.filter(None, 60, method="iir", phase="forward", picks="AUX7")
raw_stream.crop(5, None)
raw.crop(5, None)

peaks = find_peaks(
    raw.get_data(picks="AUX7").squeeze(),
    distance=0.8 * raw.info["sfreq"],
)[0]
peaks_stream = find_peaks(
    raw_stream.get_data(picks="AUX7").squeeze(),
    distance=0.8 * raw_stream.info["sfreq"],
)[0]

f, ax = plt.subplots(1, 1, layout="constrained")
ax.set_title("Respiration channel")
ax.plot(raw.times, raw.get_data(picks="AUX7").squeeze(), color="red")
ax.plot(raw_stream.times, raw_stream.get_data(picks="AUX7").squeeze())
for peak in peaks:
    ax.axvline(raw.times[peak], color="black", linestyle="--")
for peak in peaks_stream:
    ax.axvline(raw_stream.times[peak], color="teal", linestyle="--")

f, ax = plt.subplots(1, 1, layout="constrained")
ax.set_title("Peak Δ (samples)")
ax.hist(peaks_stream - peaks, bins=40)
