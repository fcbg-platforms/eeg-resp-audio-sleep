from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from byte_triggers import ParallelPortTrigger
from psychopy.sound.backend_ptb import SoundPTB

from ..utils._checks import check_type, check_value
from ._config import (
    FQ_DEVIANT,
    FQ_TARGET,
    N_DEVIANT,
    N_TARGET,
    SOUND_DURATION,
    TRIGGER_ARGS,
    TRIGGER_TYPE,
    TRIGGERS,
)

if TYPE_CHECKING:
    from byte_triggers._base import BaseTrigger


def generate_sequence() -> list[int]:
    """Generate a random sequence of target and deviant stimuli."""
    sequence = [TRIGGERS["target"]] * N_TARGET + [TRIGGERS["deviant"]] * N_DEVIANT
    rng = np.random.default_rng()
    rng.shuffle(sequence)
    return sequence


def create_trigger() -> BaseTrigger:
    """Create a trigger object.

    Returns
    -------
    trigger : Trigger
        The corresponding trigger object.
    """
    check_type(TRIGGER_TYPE, ("str",), "trigger_type")
    check_value(TRIGGER_TYPE, ("arduino", "lpt"), "TRIGGER_TYPE")
    check_type(TRIGGER_ARGS, ("str", None), "TRIGGER_ARGS")
    if TRIGGER_TYPE == "arduino":
        trigger = ParallelPortTrigger("arduino", delay=10)
    elif TRIGGER_TYPE == "lpt":
        trigger = ParallelPortTrigger(TRIGGER_ARGS, delay=10)
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
