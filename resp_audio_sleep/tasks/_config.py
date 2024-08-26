# triggers are defined in the format 'target|deviant/frequency' with frequency as float
TRIGGERS: dict[str, int] = {
    "target/1000.0": 1,
    "target/2000.0": 2,
    "target/440.0": 3,
    "deviant/1000.0": 11,
    "deviant/2000.0": 12,
}
TRIGGER_TYPE: str = "arduino"
TRIGGER_ARGS: str | None = None
TRIGGER_TASKS: dict[str, tuple[int, int]] = {
    "baseline": (200, 201),
    "synchronous-respiration": (210, 211),
    "synchronous-cardiac": (220, 221),
    "isochronous": (230, 231),
    "asynchronous": (240, 241),
}
# sound settings
N_TARGET: int = 50
N_DEVIANT: int = 0
SOUND_DURATION: float = 0.2
BLOCKSIZE: int = 4  # default 128, controls part of the latency <-> stability trade-off
# sequence and task settings
BASELINE_DURATION: float = 60  # default setting when nothing is available
EDGE_PERC: float = 10  # percentage between 0 and 100
# detector settings
ECG_HEIGHT: float = 0.985
ECG_DISTANCE: float = 0.3
RESP_PROMINENCE: float = 5
RESP_DISTANCE: float = 0.8
# target timing
TARGET_DELAY: float = 0.25
# other
INTER_BLOCK_DELAY: float = 5  # delay in seconds between blocks


# TODO: Define a configuration class to handle all configuration elements.
class ConfigRepr:  # noqa: D101
    def __repr__(self) -> str:
        """String representation of the configuration."""  # noqa: D401
        repr_str = "Configuration of the system:\n"
        repr_str += len(repr_str.strip()) * "-" + "\n"
        # triggers
        repr_str += f"Triggers:\n  type: {TRIGGER_TYPE}\n"
        if TRIGGER_ARGS is not None:
            repr_str += f"  args: {TRIGGER_ARGS}\n"
        repr_str += "  codes:\n"
        for key, value in TRIGGERS.items():
            repr_str += f"    {key}: {value}\n"
        # sounds
        repr_str += (
            f"Sounds:\n  number of targets: {N_TARGET}\n  "
            f"number of deviants: {N_DEVIANT}\n"
        )
        repr_str += f"  duration: {SOUND_DURATION} s\n"
        # sequence settings
        repr_str += f"Sequence/Task settings:\n  edge percentage: {EDGE_PERC}%\n"
        repr_str += f"  baseline duration: {BASELINE_DURATION} s\n"
        # detector settings
        repr_str += f"Detector settings:\n  ECG height: {ECG_HEIGHT}\n"
        repr_str += f"  ECG distance: {ECG_DISTANCE}\n"
        repr_str += f"  respiration distance: {RESP_DISTANCE}\n"
        return repr_str
