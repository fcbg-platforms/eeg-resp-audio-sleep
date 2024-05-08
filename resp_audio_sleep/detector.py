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


_MIN_RESP_PEAK_DISTANCE: float = 0.8  # seconds
_MIN_ECG_PEAK_DISTANCE: float = 0.3  # seconds
_RESP_PROMINENCE: float = 20  # arbitrary units
_ECG_PROMINENCE: float = 200  # arbitrary units


@fill_doc
class Detector:
    """Real-time single channel peak detector.

    Parameters
    ----------
    %(bufsize)s
    %(stream_name)s
    %(ecg_ch_name)s
    %(resp_ch_name)s
    viewer : bool
        If True, a viewer will be created to display the real-time signal and detected
        peaks. Useful for debugging, but should be set to False for production.
    """

    def __init__(
        self,
        bufsize: float,
        stream_name: str,
        ecg_ch_name: str | None,
        resp_ch_name: str | None,
        *,
        viewer: bool = False,
    ) -> None:
        if bufsize < 2:
            warn("Buffer size shorter than 2 second might be too short.")
        if ecg_ch_name is None and resp_ch_name is None:
            raise ValueError(
                "At least one of 'ecg_ch_name' or 'resp_ch_name' must be set."
            )
        check_type(viewer, (bool,), "viewer")
        self._ecg_ch_name = ecg_ch_name
        self._resp_ch_name = resp_ch_name
        self._create_stream(bufsize, stream_name)
        self._viewer = Viewer(ecg_ch_name, resp_ch_name) if viewer else None
        # peak detection settings
        self._last_peak = {"ecg": None, "resp": None}
        self._peak_candidates = {"ecg": None, "resp": None}
        self._peak_candidates_count = {"ecg": None, "resp": None}

    @fill_doc
    def _check_ch_type(self, ch_type) -> None:
        """Check validity of 'ch_type'.

        Parameters
        ----------
        %(ch_type)s
        """
        if ch_type not in ("resp", "ecg"):
            raise ValueError("The channel type must be either 'resp' or 'ecg'.")
        elif ch_type == "resp" and self._resp_ch_name is None:
            raise ValueError("No respiration channel was set.")
        elif ch_type == "ecg" and self._ecg_ch_name is None:
            raise ValueError("No ECG channel was set.")

    @fill_doc
    def _create_stream(self, bufsize: float, stream_name: str) -> None:
        """Create the LSL stream and prefill the buffer.

        Parameters
        ----------
        %(bufsize)s
        %(stream_name)s
        """
        picks = [
            elt for elt in (self._ecg_ch_name, self._resp_ch_name) if elt is not None
        ]
        self._stream = StreamLSL(bufsize, stream_name).connect(processing_flags="all")
        self._stream.pick(picks)
        self._stream.set_channel_types(
            {ch: "misc" for ch in picks}, on_unit_change="ignore"
        )
        self._stream.notch_filter(50, picks=picks)
        self._stream.notch_filter(100, picks=picks)
        logger.info("Prefilling buffer of %.2f seconds.", self._stream._bufsize)
        sleep(bufsize)
        logger.info("Buffer prefilled.")

    @fill_doc
    def _detect_peaks(self, ch_type: str) -> NDArray[np.float64]:
        """Detect all peaks in the buffer.

        Returns
        -------
        peaks : array of shape (n_peaks,)
            The timestamps of all detected peaks.
        %(ch_type)s
        """
        data, ts = self._stream.get_data(
            picks=self._resp_ch_name if ch_type == "resp" else self._ecg_ch_name
        )
        distance = (
            _MIN_RESP_PEAK_DISTANCE * self._stream.info["sfreq"]
            if ch_type == "resp"
            else _MIN_ECG_PEAK_DISTANCE * self._stream.info["sfreq"]
        )
        prominence = _RESP_PROMINENCE if ch_type == "resp" else _ECG_PROMINENCE
        peaks, _ = find_peaks(
            data.squeeze(),
            distance=distance,
            prominence=prominence,
        )
        if self._viewer is not None:
            self._viewer.plot(ts, data.squeeze(), ch_type)
        return ts[peaks]

    @fill_doc
    def new_peak(self, ch_type: str) -> float | None:
        """Detect new peak entering the buffer.

        Returns
        -------
        peak : float | None
            The timestamp of the newly detected peak. None if no new peak is detected.
        %(ch_type)s
        """
        self._check_ch_type(ch_type)
        ts_peaks = self._detect_peaks(ch_type)
        if ts_peaks.size == 0:
            return None  # unlikely to happen, but let's exit early if we have nothing
        if (
            self._peak_candidates[ch_type] is None
            and self._peak_candidates_count[ch_type] is None
        ):
            self._peak_candidates[ch_type] = list(ts_peaks)
            self._peak_candidate_counts[ch_type] = [1] * ts_peaks.size
            return None
        peaks2append = []
        for k, peak in enumerate(self._peak_candidates[ch_type]):
            if peak in ts_peaks:
                self._peak_candidate_counts[ch_type][k] += 1
            else:
                peaks2append.append(peak)
        # before going further, let's make sure we don't add too many false positives
        min_distance = (
            _MIN_RESP_PEAK_DISTANCE if ch_type == "resp" else _MIN_ECG_PEAK_DISTANCE
        )
        if int(self._stream._bufsize * (1 / min_distance)) < len(peaks2append) + len(
            self._peak_candidates[ch_type]
        ):
            self._peak_candidates[ch_type] = None
            self._peak_candidate_counts[ch_type] = None
            return None
        self._peak_candidates[ch_type].extend(peaks2append)
        self._peak_candidate_counts[ch_type].extend([1] * len(peaks2append))
        # now, all the detected peaks have been triage, let's see if we have a winner
        idx = [
            k
            for k, count in enumerate(self._peak_candidate_counts[ch_type])
            if 4 <= count
        ]
        if len(idx) == 0:
            return None
        peaks = sorted([self._peak_candidates[ch_type][k] for k in idx])
        # compare the winner with the last known peak
        if self._last_peak[ch_type] is None:  # don't return the first peak detected
            new_peak = None
            self._last_peak[ch_type] = peaks[-1]
        if (
            self._last_peak[ch_type] is None
            or self._last_peak[ch_type] + _MIN_RESP_PEAK_DISTANCE <= peaks[-1]
        ):
            new_peak = peaks[-1]
            self._last_peak[ch_type] = peaks[-1]
            if self._viewer is not None:
                self._viewer.add_peak(new_peak, ch_type)
        else:
            new_peak = None
        # reset the peak candidates
        self._peak_candidates[ch_type] = None
        self._peak_candidate_counts[ch_type] = None
        return new_peak
