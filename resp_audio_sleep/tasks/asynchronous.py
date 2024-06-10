from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import psychtoolbox as ptb

from ..utils._docs import fill_doc
from ..utils.logs import logger
from ._config import TARGET_DELAY, TRIGGERS
from ._utils import create_sounds, create_trigger, generate_sequence

if TYPE_CHECKING:
    from numpy.typing import NDArray


@fill_doc
def asynchronous(
    peaks: NDArray[np.float64],
    target: float,
    deviant: float,
) -> None:  # noqa: D401
    """Asynchronous blocks where a synchronous sequence is repeated.

    Parameters
    ----------
    peaks : array of shape (n_peaks,)
        The detected respiration peak timings in seconds.
    %(fq_target)s
    %(fq_deviant)s
    """
    # create sound stimuli, trigger and sequence
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
    # generate delays between peaks
    delays = np.diff(peaks)
    rng = np.random.default_rng()
    delays = rng.choice(delays, size=len(sequence), replace=True)
    # main loop
    counter = 0
    while counter <= len(sequence) - 1:
        wait = ptb.GetSecs() + TARGET_DELAY
        stimulus.get(sequence[counter]).play(when=wait)
        logger.debug("Triggering %i in %.2f ms.", sequence[counter], TARGET_DELAY)
        time.sleep(TARGET_DELAY)
        trigger.signal(sequence[counter])
        time.sleep(delays[counter])
        counter += 1
