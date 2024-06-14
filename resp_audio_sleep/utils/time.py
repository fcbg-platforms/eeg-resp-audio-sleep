import time


def high_precision_sleep(duration: float) -> None:
    """High precision sleep function."""
    start_time = time.perf_counter()
    assert 0 < duration, "The duration must be strictly positive."
    while True:
        elapsed_time = time.perf_counter() - start_time
        remaining_time = duration - elapsed_time
        if remaining_time <= 0:
            break
        if remaining_time >= 0.0002:
            time.sleep(remaining_time / 2)
