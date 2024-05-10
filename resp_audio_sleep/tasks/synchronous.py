import time
from typing import TYPE_CHECKING

import numpy as np
import psychtoolbox as ptb
from mne_lsl.lsl import local_clock

from ..detector import Detector
from ..utils._docs import fill_doc
from ..utils.logs import logger
from ._config import RESP_DISTANCE, RESP_PROMINENCE
from ._utils import create_sounds, create_trigger, generate_sequence

if TYPE_CHECKING:
    from numpy.typing import NDArray


@fill_doc
def synchronous(stream_name: str, resp_ch_name: str) -> NDArray[np.float64]:  # noqa: D401
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
    trigger = create_trigger("arduino")
    sequence = generate_sequence(n_target=100, n_deviant=0)
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
