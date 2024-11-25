from __future__ import annotations

from itertools import cycle

import click
import numpy as np
from psychopy import logging
from psychopy.hardware.keyboard import Keyboard
from stimuli.time import Clock, sleep

from .. import set_log_level
from ..tasks import asynchronous as asynchronous_task
from ..tasks import baseline as baseline_task
from ..tasks import isochronous as isochronous_task
from ..tasks import synchronous_cardiac as synchronous_cardiac_task
from ..tasks import synchronous_respiration as synchronous_respiration_task
from ..tasks._config import BASELINE_DURATION, INTER_BLOCK_DELAY, ConfigRepr
from ..utils.blocks import _BLOCKS, generate_blocks_sequence
from ..utils.logs import logger, warn
from ._utils import ch_name_ecg, ch_name_resp, fq_deviant, fq_target, stream, verbose
from .tasks import (
    asynchronous,
    baseline,
    isochronous,
    synchronous_cardiac,
    synchronous_respiration,
)
from .testing import (
    test_detector_cardiac,
    test_detector_respiration,
    test_sequence,
    test_triggers,
)


@click.group()
def run():
    """Entry point to start the tasks."""
    config = ConfigRepr()
    click.echo(config)


@click.command()
@click.option(
    "--n-blocks", prompt="Number of blocks", help="Number of blocks.", type=int
)
@stream
@ch_name_resp
@ch_name_ecg
@fq_target
@fq_deviant
@verbose
def paradigm(
    n_blocks: int,
    stream: str,
    ch_name_resp: str,
    ch_name_ecg: str,
    target: float,
    deviant: float,
    verbose: str,
) -> None:
    """Run the paradigm, alternating between blocks."""
    set_log_level(verbose)
    if n_blocks <= 0:
        raise ValueError(f"Number of blocks must be positive. '{n_blocks}' is invalid.")
    # prepare mapping between function and block name
    mapping_func = {
        "baseline": baseline_task,
        "isochronous": isochronous_task,
        "asynchronous": asynchronous_task,
        "synchronous-respiration": synchronous_respiration_task,
        "synchronous-cardiac": synchronous_cardiac_task,
    }
    assert len(set(mapping_func) - set(_BLOCKS)) == 0  # sanity-check
    # prepare mapping between argument and block name
    mapping_args = {
        "baseline": [BASELINE_DURATION],
        "isochronous": [None],
        "asynchronous": [None],
        "synchronous-respiration": [stream, ch_name_resp],
        "synchronous-cardiac": [stream, ch_name_ecg, None],
    }
    assert len(set(mapping_args) - set(_BLOCKS)) == 0
    # prepare mapping between keyword argument and block name, including target and
    # deviant cycling frequencis
    targets = cycle([target, deviant])
    deviants = cycle([deviant, target])
    # overwrite the same variables
    target = next(targets)
    deviant = next(deviants)
    mapping_kwargs = {
        "baseline": {},
        "isochronous": {"target": target, "deviant": deviant},
        "asynchronous": {"target": target, "deviant": deviant},
        "synchronous-respiration": {"target": target, "deviant": deviant},
        "synchronous-cardiac": {"target": target, "deviant": deviant},
    }
    assert len(set(mapping_kwargs) - set(_BLOCKS)) == 0  # sanity-check
    # create a keyboard object to monitor for breaks
    keyboard = Keyboard()
    with _disable_psychopy_logs():
        keyboard.stop()
    # execute paradigm loop
    blocks = list()
    while len(blocks) < n_blocks:
        blocks.append(generate_blocks_sequence(blocks))
        logger.info("Running block %i / %i: %s.", len(blocks), n_blocks, blocks[-1])
        clock = Clock()
        result = mapping_func[blocks[-1]](
            *mapping_args[blocks[-1]], **mapping_kwargs[blocks[-1]]
        )
        duration = clock.get_time()
        logger.info("Block '%s' took %.3f seconds.", blocks[-1], duration)
        # prepare arguments for future blocks if we just ran a respiration synchronous
        # block
        if result is not None:
            # sanity-check
            assert blocks[-1] == "synchronous-respiration"
            assert isinstance(result, np.ndarray)
            assert result.ndim == 1
            assert result.size != 0
            mapping_args["baseline"][0] = duration
            mapping_args["asynchronous"][0] = result
            mapping_args["synchronous-cardiac"][2] = result
            delay = np.median(np.diff(result))
            mapping_args["isochronous"][0] = delay
            logger.info(
                "Median delay between respiration peaks set to %.3f seconds.", delay
            )
        # prepare keyword argument for future blocks if we just ran 5 blocks
        if len(blocks) % 5 == 0:
            logger.info("Cycling target and deviant frequencies.")
            target = next(targets)
            deviant = next(deviants)
            for key, elt in mapping_kwargs.items():
                if key == "baseline":
                    continue
                elt["target"] = target
                elt["deviant"] = deviant
        # wait in the inter block delay or a space key press
        _wait_inter_block(INTER_BLOCK_DELAY, keyboard)
    logger.info("Paradigm complete. Exiting.")


def _wait_inter_block(delay: float, keyboard: Keyboard) -> None:
    """Wait the inter-block delay.

    Parameters
    ----------
    delay : float
        The delay to wait in seconds.
    keyboard : Keyboard
        The PsychoPy keyboard object used to monitor the space key press.
    """
    assert 0 < delay  # sanity-check
    clock = Clock()
    keyboard.start()
    logger.info("Inter-block for %.1f seconds (press space to pause).", delay)
    while True:
        keys = keyboard.getKeys(keyList=["space"], waitRelease=True)
        if len(keys) > 1:
            warn("Multiple space key pressed simultaneously. Skipping.")
            continue
        elif len(keys) == 1:
            logger.info("Space key pressed, pausing execution.")
            start_hold = clock.get_time_ns()
            while True:
                keys = keyboard.getKeys(keyList=["space"], waitRelease=True)
                if len(keys) > 1:
                    warn("Multiple space key pressed simultaneously. Skipping.")
                    continue
                elif len(keys) == 1:
                    break
                sleep(0.05)
            stop_hold = clock.get_time_ns()
            delay += (stop_hold - start_hold) / 1e9
            logger.info(
                "Space key pressed, resuming execution. Inter-block delay "
                "remaining duration: %.1f seconds.",
                delay - clock.get_time(),
            )
        if clock.get_time() > delay:
            break
        sleep(0.05)
    with _disable_psychopy_logs():
        keyboard.stop()
    logger.info("Inter-block complete.")


class _disable_psychopy_logs:
    def __enter__(self) -> None:
        logging.console.setLevel(logging.CRITICAL)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        logging.console.setLevel(logging.WARNING)


run.add_command(baseline)
run.add_command(isochronous)
run.add_command(asynchronous)
run.add_command(synchronous_respiration)
run.add_command(synchronous_cardiac)
run.add_command(paradigm)
run.add_command(test_detector_respiration)
run.add_command(test_detector_cardiac)
run.add_command(test_sequence)
run.add_command(test_triggers)
