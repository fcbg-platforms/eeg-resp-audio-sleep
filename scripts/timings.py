# %% Load libraries
from pathlib import Path

import numpy as np
import seaborn as sns
from mne import Epochs, find_events
from mne.io import read_raw_fif

import resp_audio_sleep

root = Path(resp_audio_sleep.__file__).parent.parent / "data" / "timings"


# %% Isochronous
fname = root / "isochronous-raw.fif"
raw = read_raw_fif(fname, preload=True)
events = find_events(raw)
epochs = Epochs(raw, events, tmin=-0.1, tmax=0.3, picks="misc")
epochs.plot(picks="AUX9", n_epochs=1, events=events, scalings=dict(misc=4e5))

# the target delay was 0.8 seconds
delays = np.diff(events[:, 0])
facetgrid = sns.displot(
    delays, kde=True, binwidth=1, binrange=(np.min(delays) - 0.5, np.max(delays) + 0.5)
)
facetgrid.fig.suptitle(
    f"Distribution of delays in samples\nmean: {np.mean(delays):.1f} - std: "
    f"{np.std(delays):.1f}"
)
facetgrid.fig.tight_layout()


# %% Asynchronous
fname = root / "asynchronous-raw.fif"
raw = read_raw_fif(fname, preload=True)
events = find_events(raw)
epochs = Epochs(raw, events, tmin=-0.1, tmax=0.3, picks="misc")
epochs.plot(picks="AUX9", n_epochs=1, events=events, scalings=dict(misc=4e5))

# the target delay was 0.5 to 1.5 seconds (uniformly distributed)
delays = np.diff(events[:, 0])
facetgrid = sns.displot(delays, kde=True)
facetgrid.fig.suptitle(
    f"Distribution of delays in samples\nmean: {np.mean(delays):.1f} - std: "
    f"{np.std(delays):.1f}"
)
facetgrid.fig.tight_layout()
