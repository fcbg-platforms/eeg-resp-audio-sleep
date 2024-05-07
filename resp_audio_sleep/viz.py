from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt

from .utils._checks import check_type

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class Viewer:
    def __init__(self, ax: plt.Axes | None = None) -> None:
        """Viewer of a real-time respiration or cardiac signal with peak detection.

        Parameters
        ----------
        ax : Axes | None
            Matplotlib axes used to draw the respiration data and peaks. If None, a new
            figure is created.
        """
        if plt.get_backend() != "QtAgg":
            plt.switch_backend("QtAgg")
        if not plt.isinteractive():
            plt.ion()  # enable interactive mode
        if ax is None:
            self._fig, self._axes = plt.subplots(1, 1, figsize=(8, 8))
        else:
            check_type(ax, (plt.Axes,), "ax")
            self._fig, self._axes = ax.get_figure(), ax
        self._peaks = []
        plt.show()

    def plot(self, ts: NDArray[np.float64], data: NDArray[np.float64]) -> None:
        """Plot the respiration or cardiac data and peaks.

        Parameters
        ----------
        ts : array of shape (n_samples,)
            Timestamps of the respiration data.
        data : array of shape (n_samples,)
            Respiration or cardiac data.
        """
        assert ts.ndim == 1
        assert data.ndim == 1
        # prune peaks outside of the viewing window
        for k, peak in enumerate(self._peaks):
            if ts[0] <= peak:
                idx = k
                break
        else:
            idx = 0
        self._peaks = self._peaks[idx:]
        # update plotting window
        self._axes.clear()
        self._axes.plot(ts, data)
        for peak in self._peaks:
            self._axes.axvline(peak, color="red", linestyle="--")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def add_peak(self, peak: float) -> None:
        """Add a peak to the viewer.

        Parameters
        ----------
        peak : float
            Timestamp of the peak.
        """
        check_type(peak, ("numeric",), "peak")
        assert 0 < peak
        self._peaks.append(peak)
