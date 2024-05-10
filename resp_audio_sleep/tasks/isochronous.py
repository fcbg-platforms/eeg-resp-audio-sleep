import time

import psychtoolbox as ptb

from ..utils._checks import check_type
from ..utils.logs import logger
from ._utils import create_sounds, create_trigger, generate_sequence


def isochronous(delay: float) -> None:  # noqa: D401
    """Isochronous auditory stimulus.

    Parameters
    ----------
    delay : float
        Delay between 2 stimuli in seconds.
    """
    check_type(delay, ("numeric",), "delay")
    if delay <= 0:
        raise ValueError("The delay must be strictly positive.")
    # create sound stimuli, trigger and sequence
    target, deviant = create_sounds()
    trigger = create_trigger()
    sequence = generate_sequence()
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
        time.sleep(delay)
        counter += 1
