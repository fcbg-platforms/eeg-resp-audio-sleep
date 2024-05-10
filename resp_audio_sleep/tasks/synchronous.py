import time
from typing import TYPE_CHECKING

import numpy as np
import psychtoolbox as ptb
from mne_lsl.lsl import local_clock

from ..detector import Detector
from ..utils._checks import check_type
from ..utils._docs import fill_doc
from ..utils.logs import logger
from ._config import ECG_DISTANCE, ECG_HEIGHT, RESP_DISTANCE, RESP_PROMINENCE
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
        wait = pos + 0.2 - local_clock()
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
        Target delay between 2 stimuli, in seconds.
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
    target = None
    while counter <= len(sequence) - 1:
        pos = detector.new_peak("ecg")
        if pos is None:
            continue
        heartrate.add_heartbeat(pos)
        if not heartrate.initialized:
            continue
        if target is not None and abs(target - pos + heartrate.mean_delay()) < abs(
            target - pos
        ):
            continue  # next r-peal will be closer from the target
        wait = pos + 0.2 - local_clock()
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
        target = pos + delay if target is None else target + delay


class _HeartRateMonitor:
    """Class to monitor the heart rate."""

    def __init__(self, size: int = 10) -> None:
        self._times = np.empty(shape=size, dtype=float)
        self._counter = 0
        self._initialized = False

    def add_heartbeat(self, pos: float) -> None:
        """Add an heartbeat measurement point."""
        self._times[self._counter] = pos
        self._counter += 1
        if self._counter == self._times.size:
            self._counter = 0
            self._initialized = True

    def mean_delay(self) -> float:
        """Mean delay between two heartbeats in seconds."""
        return np.mean(np.diff(self._times))

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
