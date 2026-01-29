r"""FPS tracking for video streaming.

Tracks frames per second using a rolling window of recent timestamps.
"""

import time
from collections import deque


class FPSTracker:
    r"""Tracks frames per second using a rolling window.

    :ivar window_size: Number of recent frames to use for FPS calculation
    :ivar current: Current FPS based on recent frame timestamps
    """

    def __init__(self, window_size: int = 30) -> None:
        r"""Initialize FPS tracker.

        :param window_size: Number of recent frames to track (default: 30)
        """
        self.window_size = window_size
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._frame_count = 0

    def record_frame(self) -> None:
        r"""Record a processed frame timestamp."""
        self._timestamps.append(time.time())
        self._frame_count += 1

    @property
    def current(self) -> float:
        r"""Get current FPS based on recent frames.

        :return: Current FPS, or 0.0 if insufficient data
        """
        if len(self._timestamps) < 2:
            return 0.0

        # Calculate FPS from time span of recent frames
        time_span = self._timestamps[-1] - self._timestamps[0]
        if time_span == 0:
            return 0.0

        return (len(self._timestamps) - 1) / time_span

    @property
    def total_frames(self) -> int:
        r"""Get total number of frames processed.

        :return: Total frame count
        """
        return self._frame_count

    def reset(self) -> None:
        r"""Reset FPS tracker to initial state."""
        self._timestamps.clear()
        self._frame_count = 0
