from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import psychtoolbox as ptb
from mne_lsl.lsl import local_clock

from ..detector import Detector
from ..utils._checks import check_type, ensure_int
from ..utils._docs import fill_doc
from ..utils.logs import logger
from ._config import (
    ECG_DISTANCE,
    ECG_HEIGHT,
    RESP_DISTANCE,
    RESP_PROMINENCE,
    TARGET_DELAY,
)
from ._utils import create_sounds, create_trigger, generate_sequence

if TYPE_CHECKING:
    from numpy.typing import NDArray


@fill_doc
def synchronous_respiration(
    stream_name: str,
    resp_ch_name: str,
) -> NDArray[np.float64]:  # noqa: D401
    """Synchronous auditory stimulus with the respiration peak signal.

    Parameters
    ----------
    %(stream_name)s
    %(resp_ch_name)s

    Returns
    -------
    peaks : array of shape (n_peaks,)
        The detected respiration peak timings in seconds.
    """
    # create sound stimuli, trigger, sequence and detector
    target, deviant = create_sounds()
    trigger = create_trigger()
    sequence = generate_sequence()
    detector = Detector(
        bufsize=4,
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
    while counter <= len(sequence) - 1:
        pos = detector.new_peak("resp")
        if pos is None:
            continue
        wait = pos + TARGET_DELAY - local_clock()
        if wait <= 0:
            logger.debug("Skipping bad detection/triggering.")
            continue
        if sequence[counter] == 1:
            target.play(when=ptb.GetSecs() + wait)
        elif sequence[counter] == 2:
            deviant.play(when=ptb.GetSecs() + wait)
        logger.debug("Triggering %i in %.3f ms.", sequence[counter], wait * 1000)
        time.sleep(wait)
        trigger.signal(sequence[counter])
        counter += 1
        peaks.append(pos)
    return np.array(peaks)


@fill_doc
def synchronous_cardiac(
    stream_name: str,
    ecg_ch_name: str,
    delay: float,
) -> None:  # noqa: D401
    """Synchronous auditory stimulus with the cardiac peak signal.

    Parameters
    ----------
    %(stream_name)s
    %(ecg_ch_name)s
    delay : float
        Target delay between 2 stimuli, in seconds. The stimulus will be synchronized
        with the cardiac peak signal, but will attempt to match the delay as closely as
        possible.
    """
    check_type(delay, ("numeric",), "delay")
    if delay <= 0:
        raise ValueError("The delay must be strictly positive.")
    # create sound stimuli, trigger, sequence and detector
    target, deviant = create_sounds()
    trigger = create_trigger()
    sequence = generate_sequence()
    detector = Detector(
        bufsize=4,
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
    while counter <= len(sequence) - 1:
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
                logger.debug(
                    "\nDecision: wait\n\tTarget: %.3f\n\tPeak: %.3f\n\tDistance to "
                    "peak: %.3f\n\tDistance to next peak: %.3f\n",
                    target_time,
                    pos,
                    distance_r_peak,
                    distance_next_r_peak,
                )
                continue  # next r-peak will be closer from the target
            logger.debug(
                "\nDecision: stim\n\tTarget: %.3f\n\tPeak: %.3f\n\tDistance to peak: "
                "%.3f\n\tDistance to next peak: %.3f\n",
                target_time,
                pos,
                distance_r_peak,
                distance_next_r_peak,
            )
        wait = pos + TARGET_DELAY - local_clock()
        if wait <= 0:
            logger.debug("Skipping bad detection/triggering.")
            continue
        if sequence[counter] == 1:
            target.play(when=ptb.GetSecs() + wait)
        elif sequence[counter] == 2:
            deviant.play(when=ptb.GetSecs() + wait)
        logger.debug("Triggering %i in %.3f ms.", sequence[counter], wait * 1000)
        time.sleep(wait)
        trigger.signal(sequence[counter])
        counter += 1
        target_time = pos + delay if target_time is None else target_time + delay


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
