import time

from ..utils._checks import check_type
from ..utils.logs import logger


def baseline(duration: float) -> None:  # noqa: D401
    """Baseline block corresponding to a resting-state recording.

    Parameters
    ----------
    duration : float
        Duration of the baseline in seconds.
    """
    check_type(duration, ("numeric",), "duration")
    if duration <= 0:
        raise ValueError("The duration must be strictly positive.")
    logger.info("Starting baseline block of %.2f seconds", duration)
    time.sleep(duration)
    logger.info("Baseline block complete.")
