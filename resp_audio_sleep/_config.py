from __future__ import annotations

from pathlib import Path

# debugging recording
RECORDER_BUFSIZE: float = 300  # in seconds
RECORDER_PATH: Path = Path.home() / "ras-data" / "debug-buffer-raw.fif"
