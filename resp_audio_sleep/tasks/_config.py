TRIGGERS: dict[str, int] = {"target": 1, "deviant": 2}
TRIGGER_TYPE: str = "arduino"
TRIGGER_ARGS: str | None = None
# sequence settings
N_TARGET: int = 100
N_DEVIANT: int = 100
# Sound settings
FQ_TARGET: int = 440
FQ_DEVIANT: int = 2000
SOUND_DURATION: float = 0.2
# detector settings
ECG_HEIGHT: float = 0.98
ECG_DISTANCE: float = 0.3
RESP_PROMINENCE: float = 20
RESP_DISTANCE: float = 0.8
