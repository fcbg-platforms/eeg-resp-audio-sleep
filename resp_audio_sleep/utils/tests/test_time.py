import time

from resp_audio_sleep.utils.time import high_precision_sleep


def test_high_precision_sleep():
    """Test high precision sleep function."""
    start = time.perf_counter()
    high_precision_sleep(0.4)
    end = time.perf_counter()
    assert 0.4 <= end - start
