from __future__ import annotations

import psychtoolbox as ptb

from ..utils._checks import check_type
from ..utils._docs import fill_doc
from ..utils.logs import logger
from ..utils.times import high_precision_sleep
from ._config import SOUND_DURATION, TARGET_DELAY, TRIGGER_TASKS, TRIGGERS
from ._utils import create_sounds, create_trigger, generate_sequence


@fill_doc
def isochronous(delay: float, *, target: float, deviant: float) -> None:  # noqa: D401
    """Isochronous auditory stimulus.

    Parameters
    ----------
    delay : float
        Delay between 2 stimuli in seconds.
    %(fq_target)s
    %(fq_deviant)s
    """
    check_type(delay, ("numeric",), "delay")
    if delay <= 0:
        raise ValueError("The delay must be strictly positive.")
    logger.info("Starting isochronous block.")
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
    # main loop
    counter = 0
    trigger.signal(TRIGGER_TASKS["isochronous"][0])
    while counter <= sequence.size - 1:
        start = ptb.GetSecs()
        stimulus.get(sequence[counter]).play(when=start + TARGET_DELAY)
        logger.debug("Triggering %i in %.2f ms.", sequence[counter], TARGET_DELAY)
        high_precision_sleep(TARGET_DELAY)
        trigger.signal(sequence[counter])
        # note that if 'delay' is too short, the value 'wait' could end up negative
        # which (1) makes no sense and (2) would raise in the sleep function.
        wait = start + delay - ptb.GetSecs()
        high_precision_sleep(wait)
        counter += 1
    # wait for the last sound to finish
    if wait < 1.1 * SOUND_DURATION:
        high_precision_sleep(1.1 * SOUND_DURATION - wait)
    trigger.signal(TRIGGER_TASKS["isochronous"][1])
    logger.info("Isochronous block complete.")
