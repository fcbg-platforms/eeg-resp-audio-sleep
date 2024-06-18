# %% Load libraries
from pathlib import Path

from matplotlib import pyplot as plt
from mne.io import read_raw_fif

import resp_audio_sleep

root = Path(resp_audio_sleep.__file__).parent.parent / "data"


# %% Compare filters
fname = root / "synchronous-respiration-raw.fif"
raw = read_raw_fif(fname, preload=True)
raw.crop(10, 19)
raw_ = raw.copy().crop(5, None)  # raw signal, unfiltered
raw.notch_filter(50, picks="AUX7", method="iir", phase="forward")
raw.notch_filter(100, picks="AUX7", method="iir", phase="forward")
raw.filter(None, 20, picks="AUX7", method="iir", phase="forward")
raw.crop(5, None)  # time for the filter without initial state to settle
data = raw.get_data(picks="AUX7").squeeze()

f, ax = plt.subplots(1, 1, layout="constrained")
ax.plot(raw.times, raw_.get_data(picks="AUX7").squeeze(), color="gray", label="raw")
ax.plot(raw.times, data, color="teal", label="filtered")
ax.legend()
