import cv2
import numpy as np
from collections import deque

# ── Tuneable constants ──────────────────────────────────────
# All thresholds are expressed as a fraction of total pixels so
# they work correctly at any camera resolution.

MOVEMENT_RATIO_THRESHOLD = 0.02   # 2% of pixels must have changed
BLUR_KERNEL_SIZE         = (5, 5) # Gaussian blur to suppress sensor noise
DIFF_THRESHOLD           = 25     # Per-pixel brightness-change minimum (0-255)

# Smoothing: liveness confirmed only if movement detected in
# at least MIN_LIVE_FRAMES out of the last WINDOW_SIZE frames.
WINDOW_SIZE      = 5
MIN_LIVE_FRAMES  = 2

# Internal rolling window — persists between calls via mutable default
_movement_history: deque = deque(maxlen=WINDOW_SIZE)


# ── Helper: validate a frame ───────────────────────────────
def _is_valid_frame(frame) -> bool:
    """Returns True only if frame is a non-empty BGR numpy array."""
    return (
        frame is not None and
        isinstance(frame, np.ndarray) and
        frame.ndim == 3 and          # must have H x W x C shape
        frame.shape[2] == 3 and      # must be 3-channel (BGR)
        frame.size > 0
    )


# ── Main function ───────────────────────────────────────────
def detect_liveness(frame, prev_frame) -> bool:
    """
    Detects whether the camera feed shows a live (moving) subject
    or a static/printed fake.

    Args:
        frame      : Current BGR frame (numpy array)
        prev_frame : Previous BGR frame (numpy array), or None

    Returns:
        True  — movement detected → likely a live person
        False — no movement       → likely a photo/screen spoof
    """
    global _movement_history

    #  Fix: validate both frames before doing anything
    if not _is_valid_frame(frame) or not _is_valid_frame(prev_frame):
        # On the very first frame prev_frame is None — don't penalise,
        # just record as "unknown" (False) and wait for the next frame.
        _movement_history.append(False)
        return _decide()

    #  Fix: resize prev_frame to match current frame if sizes differ
    if prev_frame.shape != frame.shape:
        prev_frame = cv2.resize(prev_frame, (frame.shape[1], frame.shape[0]))

    #  Fix: apply Gaussian blur to suppress sensor/compression noise
    blurred_curr = cv2.GaussianBlur(frame,      BLUR_KERNEL_SIZE, 0)
    blurred_prev = cv2.GaussianBlur(prev_frame, BLUR_KERNEL_SIZE, 0)

    # Compute per-pixel absolute difference
    diff = cv2.absdiff(blurred_prev, blurred_curr)

    # Convert to grayscale for single-channel analysis
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    #  Fix: threshold the diff so minor noise doesn't count as movement
    _, thresh = cv2.threshold(gray, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Count significantly changed pixels
    changed_pixels = cv2.countNonZero(thresh)
    total_pixels   = gray.shape[0] * gray.shape[1]   # H × W

    #  Fix: use a ratio instead of a raw pixel count so it scales
    #         with any camera resolution
    movement_ratio = changed_pixels / total_pixels
    is_moving      = movement_ratio > MOVEMENT_RATIO_THRESHOLD

    #  Fix: smooth over recent frames — a single noisy frame won't
    #         flip the liveness decision
    _movement_history.append(is_moving)
    return _decide()


def _decide() -> bool:
    """Returns True if enough recent frames show real movement."""
    live_count = sum(_movement_history)
    return live_count >= MIN_LIVE_FRAMES


# ── Optional: reset history (call between sessions/users) ──
def reset_liveness_history():
    """Clears the rolling window — useful when switching between subjects."""
    global _movement_history
    _movement_history = deque(maxlen=WINDOW_SIZE)