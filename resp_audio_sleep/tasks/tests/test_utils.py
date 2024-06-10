import pytest

from .._config import N_DEVIANT, N_TARGET
from .._utils import (
    _check_target_deviant_frequencies,
    _check_triggers,
    generate_sequence,
)


def test_check_triggers():
    """Test trigger dictionary validation."""
    _check_triggers(triggers={"target/1000": 1, "deviant/1000": 2})
    _check_triggers(triggers={"target/1000.5": 1, "deviant/1000": 2})
    with pytest.raises(TypeError, match="'trigger-key' must be an instance of str"):
        _check_triggers(triggers={1: 2})
    with pytest.raises(ValueError, match="The trigger names must be in the format"):
        _check_triggers(triggers={"targe/1000": 1, "deviant/1000": 2})
    with pytest.raises(ValueError, match="The trigger names must be in the format"):
        _check_triggers(triggers={"target/1000.0.1": 1, "deviant/1000": 2})
    with pytest.raises(ValueError, match="The trigger names must be in the format"):
        _check_triggers(triggers={"target/blabla": 1, "deviant/1000": 2})


def test_check_target_deviant_frequencies():
    """Test validation of target and deviant frequencies."""
    triggers = {"target/1000": 1, "deviant/2000": 2}
    _check_target_deviant_frequencies(1000, 2000, triggers=triggers)
    with pytest.raises(TypeError, match="must be an instance of"):
        _check_target_deviant_frequencies("1000", 2000, triggers=triggers)
    with pytest.raises(
        ValueError, match="The target frequency must be strictly positive"
    ):
        _check_target_deviant_frequencies(0, 2000, triggers=triggers)
    with pytest.raises(
        ValueError, match="The deviant frequency must be strictly positive"
    ):
        _check_target_deviant_frequencies(1000, -101, triggers=triggers)
    with pytest.raises(ValueError, match="The target frequency '2000' is not in"):
        _check_target_deviant_frequencies(2000, 1000, triggers=triggers)


def test_generate_sequence():
    """Test sequence generation."""
    if N_TARGET == 0 and N_DEVIANT == 0:
        pytest.skip("No target nor deviant stimuli.")
    sequence = generate_sequence(
        1000, 2000, triggers={"target/1000": 1, "deviant/2000": 2}
    )
    assert all(isinstance(elt, int) for elt in sequence)
    if N_TARGET != 0:
        assert sequence.count(1) == N_TARGET
    if N_DEVIANT != 0:
        assert sequence.count(2) == N_DEVIANT
