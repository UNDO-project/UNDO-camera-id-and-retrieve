"""Tests for FPS tracking."""

import time

import pytest

from src.api.websocket.fps_tracker import FPSTracker


def test_fps_tracker_initialization():
    """Test FPS tracker initializes with correct defaults."""
    tracker = FPSTracker()
    assert tracker.window_size == 30
    assert tracker.current == 0.0
    assert tracker.total_frames == 0


def test_fps_tracker_custom_window():
    """Test FPS tracker with custom window size."""
    tracker = FPSTracker(window_size=10)
    assert tracker.window_size == 10


def test_fps_tracker_single_frame():
    """Test FPS calculation with single frame returns 0."""
    tracker = FPSTracker()
    tracker.record_frame()

    assert tracker.current == 0.0
    assert tracker.total_frames == 1


def test_fps_tracker_multiple_frames():
    """Test FPS calculation with multiple frames."""
    tracker = FPSTracker(window_size=5)

    # Record frames with known timing (10 FPS)
    for _ in range(5):
        tracker.record_frame()
        time.sleep(0.1)  # 100ms between frames = 10 FPS

    fps = tracker.current
    assert 8.0 < fps < 12.0  # Allow some timing variance
    assert tracker.total_frames == 5


def test_fps_tracker_window_limit():
    """Test FPS tracker respects window size limit."""
    tracker = FPSTracker(window_size=3)

    # Record more frames than window size
    for _ in range(10):
        tracker.record_frame()
        time.sleep(0.05)

    # Should only use last 3 frames for FPS calculation
    assert tracker.total_frames == 10


def test_fps_tracker_rapid_frames():
    """Test FPS tracker with rapid frame recording."""
    tracker = FPSTracker(window_size=10)

    # Record frames without delay
    for _ in range(10):
        tracker.record_frame()

    # Should handle very high FPS
    fps = tracker.current
    assert fps > 100  # Should be very high


def test_fps_tracker_reset():
    """Test FPS tracker reset clears state."""
    tracker = FPSTracker()

    # Record some frames
    for _ in range(5):
        tracker.record_frame()
        time.sleep(0.05)

    # Reset tracker
    tracker.reset()

    assert tracker.current == 0.0
    assert tracker.total_frames == 0


def test_fps_tracker_properties():
    """Test FPS tracker properties."""
    tracker = FPSTracker()

    # Record some frames
    for _ in range(3):
        tracker.record_frame()

    # total_frames should accumulate
    assert tracker.total_frames == 3

    # current FPS should be calculable
    assert tracker.current >= 0.0


def test_fps_tracker_zero_time_span():
    """Test FPS tracker handles zero time span gracefully."""
    tracker = FPSTracker()

    # Manually add timestamps with same value (should not happen in practice)
    tracker._timestamps.append(1.0)
    tracker._timestamps.append(1.0)
    tracker._frame_count = 2

    # Should return 0 to avoid division by zero
    assert tracker.current == 0.0


@pytest.mark.parametrize(
    "window_size,expected_range",
    [
        (5, (18, 22)),  # Small window
        (10, (18, 22)),  # Medium window
        (30, (18, 22)),  # Large window
    ],
)
def test_fps_tracker_different_windows(window_size, expected_range):
    """Test FPS tracker with different window sizes."""
    tracker = FPSTracker(window_size=window_size)

    # Record frames at ~20 FPS
    for _ in range(window_size):
        tracker.record_frame()
        time.sleep(0.05)  # 50ms = 20 FPS

    fps = tracker.current
    assert expected_range[0] < fps < expected_range[1]
