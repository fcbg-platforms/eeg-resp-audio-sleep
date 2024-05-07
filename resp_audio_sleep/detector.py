from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING

import numpy as np
from mne_lsl.stream import StreamLSL
from scipy.signal import find_peaks

from .utils._checks import check_type
from .utils._docs import fill_doc
from .utils.logs import logger, warn
from .viz import Viewer

if TYPE_CHECKING:
    from numpy.typing import NDArray


_MIN_RESP_PEAK_DISTANCE: float = 0.8  # minimum distance in seconds


@fill_doc
class DetectorResp:
    """Respiration peak detector.

    Parameters
    ----------
    bufsize : float
        Size of the buffer in seconds.
    %(stream_name)s
    %(respiration_ch_name)s
    %(viewer)s
    """

    def __init__(
        self,
        bufsize: float,
        stream_name: str,
        respiration_ch_name: str,
        viewer: Viewer | None,
    ) -> None:
        if bufsize < 2:
            warn("Buffer size shorter than 2 second might be too short.")
        check_type(viewer, (Viewer, None), "viewer")
        self._stream = StreamLSL(bufsize, stream_name).connect(processing_flags="all")
        self._stream.pick(respiration_ch_name)
        self._stream.set_channel_types(
            {respiration_ch_name: "misc"}, on_unit_change="ignore"
        )
        self._stream.notch_filter(50, picks=respiration_ch_name)
        self._stream.notch_filter(100, picks=respiration_ch_name)
        # peak detection settings
        self._last_peak = None
        self._peak_candidates = None
        self._peak_candidates_count = None
        # viewer
        self._viewer = viewer

    def prefill_buffer(self) -> None:
        """Prefill an entire buffer."""
        logger.info("Prefilling buffer of %.2f seconds.", self._stream._bufsize)
        sleep(self._stream._bufsize)
        logger.info("Buffer prefilled.")

    def new_peak(self) -> float | None:
        """Detect new peak entering the buffer.

        Returns
        -------
        peak : float | None
            The timestamp of the newly detected peak. None if no new peak is detected.
        """
        ts_peaks = self.detect_peaks()
        if ts_peaks.size == 0:
            return None  # unlikely to happen, but let's exit early if we have nothing
        if self._peak_candidates is None and self._peak_candidates_count is None:
            self._peak_candidates = list(ts_peaks)
            self._peak_candidate_counts = [1] * ts_peaks.size
            return None
        peaks2append = []
        for k, peak in enumerate(self._peak_candidates):
            if peak in ts_peaks:
                self._peak_candidate_counts[k] += 1
            else:
                peaks2append.append(peak)
        # before going further, let's make sure we don't add too many false positives
        if int(self._stream._bufsize * (1 / _MIN_RESP_PEAK_DISTANCE)) < len(
            peaks2append
        ) + len(self._peak_candidates):
            self._peak_candidates = None
            self._peak_candidate_counts = None
            return None
        self._peak_candidates.extend(peaks2append)
        self._peak_candidate_counts.extend([1] * len(peaks2append))
        # now, all the detected peaks have been triage, let's see if we have a winner
        idx = [k for k, count in enumerate(self._peak_candidate_counts) if 4 <= count]
        if len(idx) == 0:
            return None
        peaks = sorted([self._peak_candidates[k] for k in idx])
        # compare the winner with the last known peak
        if self._last_peak is None:  # don't return the first peak detected
            new_peak = None
            self._last_peak = peaks[-1]
        if (
            self._last_peak is None
            or self._last_peak + _MIN_RESP_PEAK_DISTANCE <= peaks[-1]
        ):
            new_peak = peaks[-1]
            self._last_peak = peaks[-1]
            if self._viewer is not None:
                self._viewer.add_peak(new_peak)
        else:
            new_peak = None
        # reset the peak candidates
        self._peak_candidates = None
        self._peak_candidate_counts = None
        return new_peak

    def detect_peaks(self) -> NDArray[np.float64]:
        """Detect all peaks in the buffer.

        Returns
        -------
        peaks : array of shape (n_peaks,)
            The timestamps of all detected peaks.
        """
        data, ts = self._stream.get_data()
        peaks, _ = find_peaks(
            data.squeeze(),
            distance=(_MIN_RESP_PEAK_DISTANCE * self._stream.info["sfreq"]),
            prominence=20,
        )
        if self._viewer is not None:
            self._viewer.plot(ts, data.squeeze())
        return ts[peaks]


@fill_doc
class DetectorCardiac:
    """Respiration peak detector.

    Parameters
    ----------
    bufsize : float
        Size of the buffer in seconds.
    %(stream_name)s
    ecg_ch_name : str
        Name of the ECG channel on the LSL stream.
    %(viewer)s
    """

    def __init__(
        self,
        bufsize: float,
        stream_name: str,
        ecg_ch_name: str,
        viewer: Viewer | None,
    ) -> None:
        if bufsize < 2:
            warn("Buffer size shorter than 2 second might be too short.")
        check_type(viewer, (Viewer, None), "viewer")
        self._stream = StreamLSL(bufsize, stream_name).connect(processing_flags="all")
        self._stream.pick(ecg_ch_name)
        self._stream.set_channel_types({ecg_ch_name: "ecg"}, on_unit_change="ignore")
        self._stream.notch_filter(50, picks=ecg_ch_name)
        self._stream.notch_filter(100, picks=ecg_ch_name)
        # peak detection settings
        self._last_peak = None
        self._peak_candidates = None
        self._peak_candidates_count = None
        # viewer
        self._viewer = viewer
