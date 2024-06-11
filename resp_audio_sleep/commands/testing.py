from __future__ import annotations

import time

import click
import numpy as np

from .. import set_log_level
from ..tasks._config import TRIGGERS
from ..tasks._utils import create_trigger, generate_sequence
from ._utils import fq_deviant, fq_target, verbose


@click.command()
def test_detector() -> None:
    """Test the detector settings."""


@click.command()
@fq_target
@fq_deviant
def test_sequence(target: float, deviant: float, verbose: str) -> None:
    """Test the sequence generation settings."""
    from matplotlib import pyplot as plt

    set_log_level(verbose)
    sequence = generate_sequence(target, deviant)
    trigger_target = TRIGGERS[f"target/{target}"]
    trigger_deviant = TRIGGERS[f"deviant/{deviant}"]
    f, ax = plt.subplots(1, 1, layout="constrained")
    idx = np.where(sequence == trigger_target)[0]
    ax.scatter(idx, np.ones(idx.size) * trigger_target, color="teal", label="target")
    idx = np.where(sequence == trigger_deviant)[0]
    ax.scatter(idx, np.ones(idx.size) * trigger_deviant, color="coral", label="deviant")
    ax.set_xlabel("Stimulus number")
    ax.set_ylabel("Trigger")
    ax.set_title("Sequence of stimuli")
    ax.legend()
    plt.show(block=True)


@click.command()
@verbose
def test_triggers(verbose) -> None:
    """Test the trigger settings."""
    set_log_level(verbose)
    trigger = create_trigger()
    for key, value in TRIGGERS.items():
        click.echo(f"Trigger {key}: {value}")
        trigger.signal(value)
        time.sleep(0.5)
