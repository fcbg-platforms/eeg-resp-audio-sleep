from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from byte_triggers import ParallelPortTrigger
from psychopy.sound.backend_ptb import SoundPTB

from ..utils._checks import check_type, check_value, ensure_int
from ..utils._docs import fill_doc
from ..utils.logs import logger
from ._config import (
    N_DEVIANT,
    N_TARGET,
    SOUND_DURATION,
    TRIGGER_ARGS,
    TRIGGER_TYPE,
    TRIGGERS,
)

if TYPE_CHECKING:
    from byte_triggers._base import BaseTrigger


@fill_doc
def _check_triggers(*, triggers: dict[str, int] = TRIGGERS) -> None:
    """Check that the trigger dictionary is correctly formatted.

    Parameters
    ----------
    %(triggers_dict)s
    """
    pattern = re.compile(r"^\b(target|deviant)\b\/\d+(\.\d+)?$")
    for elt in triggers:
        check_type(elt, (str,), "trigger-key")
        if not re.fullmatch(pattern, elt):
            raise ValueError(
                "The trigger names must be in the format 'name/frequency', "
                f"but got '{elt}', with name set to 'target' or 'deviant'."
            )


@fill_doc
def _check_target_deviant_frequencies(
    target: float, deviant: float, *, triggers: dict[str, int] = TRIGGERS
) -> None:
    """Check that the target and deviant frequencies are valid.

    Parameters
    ----------
    %(fq_target)s
    %(fq_deviant)s
    %(triggers_dict)s
    """
    _check_triggers()
    for name, value in zip(("target", "deviant"), (target, deviant), strict=True):
        check_type(value, ("numeric",), name)
        if value <= 0:
            raise ValueError(
                f"The {name} frequency must be strictly positive. Provided {value} is "
                "invalid."
            )
        if f"{name}/{value}" not in triggers:
            raise ValueError(
                f"The {name} frequency '{value}' is not in the trigger dictionary."
            )


@fill_doc
def create_sounds(*, triggers: dict[str, int] = TRIGGERS) -> dict[str, SoundPTB]:
    """Create auditory simuli.

    Parameters
    ----------
    %(triggers_dict)s

    Returns
    -------
    sounds : dict
        The sounds to use in the task, with the keys as sound frequency (str) and the
        values as the corresponding SoundPTB object.
    """
    _check_triggers(triggers=triggers)
    frequencies = set(elt.split("/")[1] for elt in triggers)
    return {
        frequency: SoundPTB(value=float(frequency), secs=SOUND_DURATION, blockSize=128)
        for frequency in frequencies
    }


def create_trigger() -> BaseTrigger:
    """Create a trigger object.

    Returns
    -------
    trigger : Trigger
        The corresponding trigger object.
    """
    check_type(TRIGGER_TYPE, (str,), "trigger_type")
    check_value(TRIGGER_TYPE, ("arduino", "lpt"), "TRIGGER_TYPE")
    check_type(TRIGGER_ARGS, (str, None), "TRIGGER_ARGS")
    if TRIGGER_TYPE == "arduino":
        trigger = ParallelPortTrigger("arduino", delay=10)
    elif TRIGGER_TYPE == "lpt":
        trigger = ParallelPortTrigger(TRIGGER_ARGS, delay=10)
    return trigger


@fill_doc
def generate_sequence(
    target: float, deviant: float, *, triggers: dict[str, int] = TRIGGERS
) -> list[int]:
    """Generate a random sequence of target and deviant stimuli.

    Parameters
    ----------
    %(fq_target)s
    %(fq_deviant)s
    %(triggers_dict)s

    Returns
    -------
    sequence : list
        The sequence of stimuli, with the target and deviant sounds randomly ordered.
    """
    n_target = ensure_int(N_TARGET, "N_TARGET")
    n_deviant = ensure_int(N_DEVIANT, "N_DEVIANT")
    _check_target_deviant_frequencies(target, deviant, triggers=triggers)
    trigger_target = triggers[f"target/{target}"]
    trigger_deviant = triggers[f"deviant/{deviant}"]
    logger.debug(
        "Generating a sequence of %i target and %i deviant stimuli, using %s for "
        "target and %s for deviant.",
        n_target,
        n_deviant,
        trigger_target,
        trigger_deviant,
    )
    sequence = [trigger_target] * n_target + [trigger_deviant] * n_deviant
    rng = np.random.default_rng()
    rng.shuffle(sequence)
    return sequence
