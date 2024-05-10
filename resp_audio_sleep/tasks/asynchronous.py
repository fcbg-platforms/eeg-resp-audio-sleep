from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import psychtoolbox as ptb

from ..utils.logs import logger
from ._utils import create_sounds, create_trigger, generate_sequence

if TYPE_CHECKING:
    from numpy.typing import NDArray


def asynchronous(peaks: NDArray[np.float64]) -> None:  # noqa: D401
    """Asynchronous blocks where a synchronous sequence is repeated.

    Parameters
    ----------
    peaks : array of shape (n_peaks,)
        The detected respiration peak timings in seconds.
    """
    # create sound stimuli, trigger and sequence
    target, deviant = create_sounds()
    trigger = create_trigger()
    sequence = generate_sequence()
    # generate delays between peaks
    delays = np.diff(peaks)
    rng = np.random.default_rng()
    delays = rng.choice(delays, size=len(sequence), replace=True)
    # main loop
    counter = 0
    while counter <= len(sequence) - 1:
        wait = ptb.GetSecs() + 0.2
        if sequence[counter] == 1:
            target.play(when=wait)
        elif sequence[counter] == 2:
            deviant.play(when=wait)
        logger.debug("Triggering %i in 200 ms.", sequence[counter])
        time.sleep(0.2)
        trigger.signal(sequence[counter])
        time.sleep(delays[counter])
        counter += 1
