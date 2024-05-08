import time

import psychtoolbox as ptb
from byte_triggers import ParallelPortTrigger
from mne_lsl.lsl import local_clock
from psychopy.sound.backend_ptb import SoundPTB

from ..detector import Detector
from ..utils._docs import fill_doc
from ..utils.logs import logger
from ._config import TRIGGERS
from ._utils import generate_sequence


@fill_doc
def synchronous(stream_name: str, resp_ch_name: str):  # noqa: D401
    """Synchronous auditory stimulus with the respiration peak signal.

    Parameters
    ----------
    %(stream_name)s
    %(resp_ch_name)s
    """
    # create sound stimuli and trigger
    target = SoundPTB(value=440, secs=0.2, blockSize=128)
    deviant = SoundPTB(value=2000, secs=0.2, blockSize=128)
    assert "target" in TRIGGERS  # sanity-check
    assert "deviant" in TRIGGERS  # sanity-check
    trigger = ParallelPortTrigger("arduino", delay=10)
    # create detector and sequences of events
    sequence = generate_sequence(n_target=100, n_deviant=0)
    detector = Detector(
        bufsize=4,
        stream_name=stream_name,
        ecg_ch_name=None,
        resp_ch_name=resp_ch_name,
        ecg_height=None,
        ecg_distance=None,
        resp_prominence=20,
        resp_distance=0.8,
        viewer=False,
    )
    # main loop
    counter = 0
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
