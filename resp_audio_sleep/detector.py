from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING

import numpy as np
from mne_lsl.stream import StreamLSL
from scipy.signal import find_peaks

from .utils.logs import logger, warn

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Detector:
    def __init__(
        self, bufsize: float, stream_name: str, respiration_ch_name: str
    ) -> None:
        if bufsize < 2:
            warn("Buffer size shorter than 2 second might be too short.")
        self._stream = StreamLSL(bufsize, stream_name).connect(processing_flags="all")
        self._stream.pick(respiration_ch_name)
        self._stream.set_channel_types({respiration_ch_name: "misc"})
        self._stream.notch_filter(50, picks=respiration_ch_name)
        self._stream.filter(0.1, 5, picks=respiration_ch_name)
        # peak detection settings
        self._last_peak = None

    def prefill_buffer(self) -> None:
        """Prefill an entire buffer."""
        logger.info("Prefilling buffer of %.2f seconds.", self._stream._bufsize)
        sleep(self._stream._bufsize)
        logger.info("Buffer prefilled.")

    def new_peak(self) -> float | None:
        """Detect new peak entering the buffer."""
        ts_peaks = self.detect_peaks()
        if ts_peaks.size == 0:
            return None  # unlikely to happen, but let's exit early if we have nothing
        if self._last_peak is None:  # first peak to be detected
            self._last_peak = ts_peaks[-1]
            return ts_peaks[-1]
        if ts_peaks[-1] == self._last_peak:  # already found this peak
            return None
        elif ts_peaks[-1] - self._last_peak <= 0.5:
            logger.debug("Two peaks detected too close to each other.")
            return None
        else:
            self._last_peak = ts_peaks[-1]
            return ts_peaks[-1]

    def detect_peaks(self) -> NDArray[np.float64]:
        """Detects all peaks in the buffer."""
        data, ts = self._stream.get_data()
        peaks, _ = find_peaks(data.squeeze(), height=10)
        return ts[peaks]
