import time

import psychtoolbox as ptb
from byte_triggers import ParallelPortTrigger
from mne_lsl.lsl import local_clock
from psychopy.sound import SoundPTB

from ..detector import DetectorResp
from ..utils._docs import fill_doc
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
    target = SoundPTB(value=1000, secs=0.2, blockSize=128)
    deviant = SoundPTB(value=2000, secs=0.2, blockSize=128)
    assert "target" in TRIGGERS
    assert "deviant" in TRIGGERS
    trigger = ParallelPortTrigger("arduino")
    # create detector and sequences of events
    sequence = generate_sequence(n_target=30, n_deviant=10)
    detector = DetectorResp(
        bufsize=4,
        stream_name=stream_name,
        ecg_ch_name=None,
        resp_ch_name=resp_ch_name,
        viewer=False,
    )
    # main loop
    counter = 0
    while counter <= len(sequence) - 1:
        pos = detector.new_peak()
        if pos is None:
            continue
        wait = pos + 0.2 - local_clock()
        if sequence[counter] == 1:
            target.play(when=ptb.GetSecs() + wait)
        elif sequence[counter] == 2:
            deviant.play(when=ptb.GetSecs() + wait)
        time.sleep(wait)
        trigger.signal(sequence[counter])
        counter += 1
