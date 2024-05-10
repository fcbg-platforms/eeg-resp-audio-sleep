from typing import TYPE_CHECKING

import numpy as np
from byte_triggers import ParallelPortTrigger
from psychopy.sound.backend_ptb import SoundPTB

from ..utils._checks import check_type, check_value, ensure_int
from ._config import FQ_DEVIANT, FQ_TARGET, SOUND_DURATION, TRIGGERS

if TYPE_CHECKING:
    from byte_triggers._base import BaseTrigger


def generate_sequence(n_target: int, n_deviant: int) -> list[int]:
    """Generate a random sequence of target and deviant stimuli."""
    n_target = ensure_int(n_target, "n_target")
    n_deviant = ensure_int(n_deviant, "n_deviant")
    sequence = [TRIGGERS["target"]] * n_target + [TRIGGERS["deviant"]] * n_deviant
    rng = np.random.default_rng()
    rng.shuffle(sequence)
    return sequence


def create_trigger(trigger_type: str, trigger_args: str | None = None) -> BaseTrigger:
    """Create a trigger object.

    Parameters
    ----------
    trigger_type : str
        The type of trigger to create. One of 'arduino' or 'lpt'.
    trigger_args : str | None
        Argument to pass to the trigger constructor, if any.
        - 'arduino': None
        - 'lpt': the address of the parallel port.

    Returns
    -------
    trigger : Trigger
        The corresponding trigger object.
    """
    check_type(trigger_type, ("str",), "trigger_type")
    check_value(trigger_type, ("arduino", "lpt"), "trigger_type")
    check_type(trigger_args, ("str", None), "trigger_args")
    if trigger_type == "arduino":
        trigger = ParallelPortTrigger("arduino", delay=10)
    elif trigger_type == "lpt":
        trigger = ParallelPortTrigger(trigger_args, delay=10)
    return trigger


def create_sounds() -> tuple[SoundPTB, SoundPTB]:
    """Create auditory simuli.

    Returns
    -------
    target : Sound
        The target psychopy sound object.
    deviant : Sound
        The deviant psychopy sound object.
    """
    target = SoundPTB(value=FQ_TARGET, secs=SOUND_DURATION, blockSize=128)
    deviant = SoundPTB(value=FQ_DEVIANT, secs=SOUND_DURATION, blockSize=128)
    return target, deviant
