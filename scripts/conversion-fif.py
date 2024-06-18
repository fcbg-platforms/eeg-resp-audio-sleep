from pathlib import Path

from mne import create_info
from mne.io import RawArray
from pyxdf import load_xdf

root = Path(r"/home/scheltie/Documents/CurrentStudy")
fname = "synchronous-respiration-raw.xdf"
stream = load_xdf(root / fname)[0][0]
ch_names = [
    ch["label"][0] for ch in stream["info"]["desc"][0]["channels"][0]["channel"]
]
info = create_info(ch_names, 1024, "eeg")
raw = RawArray(stream["time_series"].T, info)
raw.pick(("TRIGGER", "AUX7", "AUX8", "AUX9"))
raw.set_channel_types(
    {"TRIGGER": "stim", "AUX7": "misc", "AUX8": "misc", "AUX9": "misc"}
)
raw.save((root / fname).with_suffix(".fif"), overwrite=True)
