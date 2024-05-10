TRIGGERS: dict[str, int] = {"target": 1, "deviant": 2}
TRIGGER_TYPE: str = "arduino"
TRIGGER_ARGS: str | None = None
# sequence settings
N_TARGET: int = 25
N_DEVIANT: int = 0
# sound settings
FQ_TARGET: float = 440.0
FQ_DEVIANT: float = 2000.0
SOUND_DURATION: float = 0.2
# detector settings
ECG_HEIGHT: float = 0.99
ECG_DISTANCE: float = 0.3
RESP_PROMINENCE: float = 20
RESP_DISTANCE: float = 0.8
# target timing
TARGET_DELAY: float = 0.2
