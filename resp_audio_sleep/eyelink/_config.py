from pathlib import Path
from typing import Any

# PsychoPy windows constants
SCREEN = 1
SCREEN_KWARGS: dict[str, Any] = dict(
    allowGUI=False,
    color=(-1, -1, -1),
    fullscr=True,
    monitor=None,
    winType="pyglet",
)

# Eye-tracker constants
FOREGROUND_COLOR = (0, 0, 0)
HOST_IP: str = "100.1.1.1"
DATA_FOLDER_PATH: Path = Path.home() / "Documents" / "ras-data" / "eyelink"
