from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import psychtoolbox as ptb
from mne_lsl.lsl import local_clock

from ..detector import Detector
from ..utils._checks import check_type, ensure_int
from ..utils._docs import fill_doc
from ..utils.logs import logger
from ..utils.time import high_precision_sleep
from ._config import (
    ECG_DISTANCE,
    ECG_HEIGHT,
    RESP_DISTANCE,
    RESP_PROMINENCE,
    SOUND_DURATION,
    TARGET_DELAY,
    TRIGGER_TASKS,
    TRIGGERS,
)
from ._utils import create_sounds, create_trigger, generate_sequence

if TYPE_CHECKING:
    from byte_triggers._base import BaseTrigger
    from numpy.typing import NDArray
    from psychopy.sound.backend_ptb import SoundPTB


@fill_doc
def synchronous_respiration(
    stream_name: str,
    resp_ch_name: str,
    *,
    target: float,
    deviant: float,
) -> NDArray[np.float64]:
    """Synchronous auditory stimulus with the respiration peak signal.

    Parameters
    ----------
    %(stream_name)s
    %(resp_ch_name)s
    %(fq_target)s
    %(fq_deviant)s

    Returns
    -------
    peaks : array of shape (n_peaks,)
        The detected respiration peak timings in seconds.
    """  # noqa: D401
    logger.info("Starting respiration synchronous block.")
    # create sound stimuli, trigger, sequence
    sounds = create_sounds()
    trigger = create_trigger()
    sequence = generate_sequence(target, deviant)
    # the sequence, sound and trigger generation validates the trigger dictionary, thus
    # we can safely map the target and deviant frequencies to their corresponding
    # trigger values and sounds.
    stimulus = {
        TRIGGERS[f"target/{target}"]: sounds[str(target)],
        TRIGGERS[f"deviant/{deviant}"]: sounds[str(deviant)],
    }
    # create detector
    detector = Detector(
        stream_name=stream_name,
        ecg_ch_name=None,
        resp_ch_name=resp_ch_name,
        ecg_height=None,
        ecg_distance=None,
        resp_prominence=RESP_PROMINENCE,
        resp_distance=RESP_DISTANCE,
        viewer=False,
    )
    # main loop
    counter = 0
    peaks = []
    trigger.signal(TRIGGER_TASKS["synchronous-respiration"][0])
    while counter <= sequence.size - 1:
        pos = detector.new_peak("resp")
        if pos is None:
            continue
        success = _deliver_stimuli(pos, sequence[counter], stimulus, trigger)
        if not success:
            continue
        counter += 1
        peaks.append(pos)
    # wait for the last sound to finish
    high_precision_sleep(1.1 * SOUND_DURATION)
    trigger.signal(TRIGGER_TASKS["synchronous-respiration"][1])
    logger.info("Respiration synchronous block complete.")
    return np.array(peaks)


@fill_doc
def synchronous_cardiac(
    stream_name: str,
    ecg_ch_name: str,
    peaks: NDArray[np.float64],
    *,
    target: float,
    deviant: float,
) -> None:
    """Synchronous auditory stimulus with the cardiac peak signal.

    Parameters
    ----------
    %(stream_name)s
    %(ecg_ch_name)s
    %(peaks)s
    %(fq_target)s
    %(fq_deviant)s
    """  # noqa: D401
    check_type(peaks, (np.ndarray,), "peaks")
    if peaks.ndim != 1:
        raise ValueError("The peaks array must be one-dimensional.")
    logger.info("Starting cardiac synchronous block.")
    # create sound stimuli, trigger, sequence
    sounds = create_sounds()
    trigger = create_trigger()
    sequence = generate_sequence(target, deviant)
    # the sequence, sound and trigger generation validates the trigger dictionary, thus
    # we can safely map the target and deviant frequencies to their corresponding
    # trigger values and sounds.
    stimulus = {
        TRIGGERS[f"target/{target}"]: sounds[str(target)],
        TRIGGERS[f"deviant/{deviant}"]: sounds[str(deviant)],
    }
    # generate delays between peaks and rng to select delays
    rng = np.random.default_rng()
    delays = np.diff(peaks)
    edges = np.percentile(delays, [10, 90])
    delays = delays[np.where((edges[0] < delays) & (delays < edges[1]))]
    delays = rng.choice(delays, size=sequence.size, replace=True)
    # create detector
    detector = Detector(
        stream_name=stream_name,
        ecg_ch_name=ecg_ch_name,
        resp_ch_name=None,
        ecg_height=ECG_HEIGHT,
        ecg_distance=ECG_DISTANCE,
        resp_prominence=None,
        resp_distance=None,
        viewer=False,
    )
    # create heart-rate monitor
    heartrate = _HeartRateMonitor()
    # main loop
    counter = 0
    target_time = None
    last_pos = None
    trigger.signal(TRIGGER_TASKS["synchronous-cardiac"][0])
    while counter <= sequence.size - 1:
        pos = detector.new_peak("ecg")
        if pos is None:
            continue
        heartrate.add_heartbeat(pos)
        if not heartrate.initialized:
            continue
        if target_time is not None:
            distance_r_peak = abs(pos - target_time)
            distance_next_r_peak = abs(target_time - (pos + heartrate.mean_delay()))
            if distance_next_r_peak < distance_r_peak:
                continue  # next r-peak will be closer from the target
        success = _deliver_stimuli(pos, sequence[counter], stimulus, trigger)
        if not success:
            continue
        counter += 1
        # figure out what our next target time should be, based on the delays in the
        # previous synchronous respiration block and based on the last triggered
        # R-peak.
        if target_time is None:
            mask = np.zeros(delays.size, dtype=bool)
            mask[0] = True
            rng.shuffle(mask)
        else:
            # look for the closest delay to the last R-peak
            idx = np.argmin(np.abs(delays - (pos - last_pos)))
            mask = np.zeros(delays.size, dtype=bool)
            mask[idx] = True
        target_time = pos + delays[mask][0]
        delays = delays[~mask]
        last_pos = pos
    high_precision_sleep(1.1 * SOUND_DURATION)
    trigger.signal(TRIGGER_TASKS["synchronous-cardiac"][1])
    logger.info("Cardiac synchronous block complete.")


class _HeartRateMonitor:
    """Class to monitor the heart rate."""

    def __init__(self, size: int = 10) -> None:
        self._times = np.empty(shape=ensure_int(size, "size"), dtype=float)
        self._counter = 0
        self._initialized = False

    def add_heartbeat(self, pos: float) -> None:
        """Add an heartbeat measurement point."""
        self._times = np.roll(self._times, shift=-1)
        self._times[-1] = pos
        self._counter += 1
        if self._counter == self._times.size:
            logger.info("Heart-rate monitor initialized.")
            self._initialized = True

    def mean_delay(self) -> float:
        """Mean delay between two heartbeats in seconds."""
        if not self._initialized:
            raise ValueError("The monitor is not initialized yet.")
        mean_delay = np.mean(np.diff(self._times))
        logger.debug("Mean delay between heartbeats: %.3f s.", mean_delay)
        return mean_delay

    def rate(self) -> float:
        """Heart rate in beats per second, i.e. Hz."""
        return 1 / self.mean_delay()

    def bpm(self) -> float:
        """Heart rate in beats per minute."""
        return self.rate() * 60

    @property
    def initialized(self) -> bool:
        """Whether the monitor is initialized."""
        return self._initialized


def _deliver_stimuli(
    pos: float, elt: int, stimulus: dict[int, SoundPTB], trigger: BaseTrigger
) -> bool:
    """Deliver precisely a sound and its trigger."""
    wait = pos + TARGET_DELAY - local_clock()
    if wait <= 0:
        logger.info(
            "Skipping bad detection/triggering, too late by %.3f ms.", -wait * 1000
        )
        return False
    stimulus.get(elt).play(when=ptb.GetSecs() + wait)
    logger.debug("Triggering %i in %.3f ms.", elt, wait * 1000)
    high_precision_sleep(wait)
    trigger.signal(elt)
    return True
