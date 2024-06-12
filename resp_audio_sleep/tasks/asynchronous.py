from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import psychtoolbox as ptb

from ..utils._checks import check_type
from ..utils._docs import fill_doc
from ..utils.logs import logger
from ._config import SOUND_DURATION, TARGET_DELAY, TRIGGER_TASKS, TRIGGERS
from ._utils import create_sounds, create_trigger, generate_sequence

if TYPE_CHECKING:
    from numpy.typing import NDArray


@fill_doc
def asynchronous(
    peaks: NDArray[np.float64],
    *,
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
    check_type(peaks, (np.ndarray,), "peaks")
    if peaks.ndim != 1:
        raise ValueError("The peaks array must be one-dimensional.")
    logger.info("Starting asynchronous block.")
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
    delays = rng.choice(delays, size=sequence.size, replace=True)
    # main loop
    counter = 0
    trigger.signal(TRIGGER_TASKS["asynchronous"][0])
    while counter <= sequence.size - 1:
        start = ptb.GetSecs()
        stimulus.get(sequence[counter]).play(when=start + TARGET_DELAY)
        logger.debug("Triggering %i in %.2f ms.", sequence[counter], TARGET_DELAY)
        time.sleep(TARGET_DELAY)
        trigger.signal(sequence[counter])
        wait = start + delays[counter] - ptb.GetSecs()
        time.sleep(wait)
        counter += 1
    # wait for the last sound to finish
    if wait < 1.1 * SOUND_DURATION:
        time.sleep(wait - 1.1 * SOUND_DURATION)
    trigger.signal(TRIGGER_TASKS["asynchronous"][1])
    logger.info("Asynchronous block complete.")
