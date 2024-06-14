from byte_triggers import ParallelPortTrigger

from resp_audio_sleep.detector import Detector

trigger = ParallelPortTrigger("arduino", delay=10)
stream_name = "eegoSports 000325"

# %% Simultaneous detection of ECG and RESP peaks
detector = Detector(
    4,
    stream_name,
    ecg_ch_name="AUX8",
    resp_ch_name="AUX7",
    ecg_height=0.98,
    ecg_distance=0.2,
    resp_distance=0.8,
    viewer=False,
)

counter = 0
while counter <= 100:
    peak = detector.new_peak("ecg")
    if peak is not None:
        trigger.signal(1)
        counter += 1
    peak = detector.new_peak("resp")
    if peak is not None:
        trigger.signal(2)
        counter += 1

# %% Respiration-only detector
detector = Detector(
    4,
    stream_name,
    ecg_ch_name=None,
    resp_ch_name="AUX7",
    ecg_height=None,
    ecg_distance=None,
    resp_distance=0.8,
    viewer=False,
)

counter = 0
while counter <= 20:
    peak = detector.new_peak("resp")
    if peak is not None:
        trigger.signal(2)
        counter += 1

# %% ECG-only detector
detector = Detector(
    4,
    stream_name,
    ecg_ch_name="AUX8",
    resp_ch_name=None,
    ecg_height=0.98,
    ecg_distance=0.2,
    resp_distance=None,
    viewer=False,
)

counter = 0
while counter <= 60:
    peak = detector.new_peak("ecg")
    if peak is not None:
        trigger.signal(1)
        counter += 1
