TRIGGERS: dict[str, int] = {
    "target/1000": 1,
    "target/2000": 2,
    "deviant/1000": 11,
    "deviant/2000": 12,
}
TRIGGER_TYPE: str = "arduino"
TRIGGER_ARGS: str | None = None
# sound settings
N_TARGET: int = 25
N_DEVIANT: int = 5
SOUND_DURATION: float = 0.2
EDGE_PERC: float = 10  # percentage between 0 and 100
# detector settings
ECG_HEIGHT: float = 0.99
ECG_DISTANCE: float = 0.3
RESP_PROMINENCE: float = 20
RESP_DISTANCE: float = 0.8
# target timing
TARGET_DELAY: float = 0.2
