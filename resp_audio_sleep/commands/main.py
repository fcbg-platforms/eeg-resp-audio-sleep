import click

from .tasks import (
    asynchronous,
    baseline,
    isochronous,
    synchronous_cardiac,
    synchronous_respiration,
)


@click.group()
def run():
    """Entry point to start the tasks."""


run.add_command(baseline)
run.add_command(isochronous)
run.add_command(asynchronous)
run.add_command(synchronous_respiration)
run.add_command(synchronous_cardiac)
