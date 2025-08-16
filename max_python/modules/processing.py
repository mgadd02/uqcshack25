from typing import Tuple, Optional
import numpy as np
import cv2

def extract_play_area(frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Very simple heuristic:
    - Assume the top ~45% contains a bright/white play area rectangle.
    - Detect the largest high-brightness contour there and crop it.
    Returns (crop, (x,y,w,h)).
    """
    if frame is None:
        raise ValueError("No frame")

    h, w, _ = frame.shape
    roi_h = int(h * 0.55)
    roi = frame[:roi_h, :, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    # Threshold for bright (tune if needed)
    _, th = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    th = cv2.medianBlur(th, 5)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: return the top half
        return roi.copy(), (0, 0, w, roi_h)

    biggest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(biggest)

    # Expand a bit
    pad = 4
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(roi_h, y + bh + pad)

    crop = frame[y0:y1, x0:x1, :].copy()
    return crop, (x0, y0, x1 - x0, y1 - y0)

def detect_players(play: np.ndarray) -> dict:
    """
    Placeholder: detect left/right 'players' by bright blobs near edges.
    Returns dict with lists of coordinates (for demo).
    """
    gray = cv2.cvtColor(play, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    left_pts, right_pts = [], []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        cx = x + bw // 2
        cy = y + bh // 2
        if cx < w * 0.33:
            left_pts.append((cx, cy))
        elif cx > w * 0.66:
            right_pts.append((cx, cy))

    return {
        "left_team": left_pts,
        "right_team": right_pts,
        "center": [(x + bw // 2, y + bh // 2) for (x, y, bw, bh) in [cv2.boundingRect(c) for c in contours]]
    }
