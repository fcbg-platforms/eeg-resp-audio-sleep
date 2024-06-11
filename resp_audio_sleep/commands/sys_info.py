from __future__ import annotations

import click

from .. import sys_info as sys_info_function


@click.command()
@click.option(
    "--developer",
    help="display information for optional dependencies",
    is_flag=True,
)
def sys_info(developer: bool) -> None:
    """Run sys_info() command."""
    sys_info_function(developer=developer)
