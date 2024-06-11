import click

fq_deviant = click.option(
    "--deviant",
    prompt="Deviant frequency (Hz)",
    help="Frequency of the deviant stimulus in Hz.",
    type=float,
)
fq_target = click.option(
    "--target",
    prompt="Target frequency (Hz)",
    help="Frequency of the target stimulus in Hz.",
    type=float,
)
stream = click.option(
    "--stream",
    prompt="LSL stream name",
    help="Name of the stream to use for the synchronous task.",
    type=str,
)
verbose = click.option(
    "--verbose",
    help="Verbosity level.",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
    default="INFO",
    show_default=True,
)
