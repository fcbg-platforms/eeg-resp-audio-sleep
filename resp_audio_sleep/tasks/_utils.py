import numpy as np

from ..utils._checks import ensure_int
from ._config import TRIGGERS


def generate_sequence(n_target: int, n_deviant: int) -> list[int]:
    """Generate a random sequence of target and deviant stimuli."""
    n_target = ensure_int(n_target, "n_target")
    n_deviant = ensure_int(n_deviant, "n_deviant")
    sequence = [TRIGGERS["target"]] * n_target + [TRIGGERS["deviant"]] * n_deviant
    rng = np.random.default_rng()
    rng.shuffle(sequence)
    return sequence
