"""
Utility functions for the face detection system.
"""
import os
import time
from collections import deque

import cv2


def convert_rgb_to_bgr(image):
    """Convert RGB image to BGR (for OpenCV)."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def convert_bgr_to_rgb(image):
    """Convert BGR image to RGB (for face_recognition and Pillow)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_face_box(image, location, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box around a face.

    Args:
        image: Image to draw on (modified in place)
        location: Face location (top, right, bottom, left)
        color: Box color (BGR)
        thickness: Line thickness

    Returns:
        Image with drawn box
    """
    top, right, bottom, left = location
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
    return image


def save_image(image, filepath):
    """Save a BGR image, creating the parent folder if needed. Returns the path."""
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)
    cv2.imwrite(filepath, image)
    return filepath


def load_image(filepath):
    """
    Load image from file.

    Args:
        filepath: Path to image file

    Returns:
        Image as numpy array (RGB)
    """
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Could not load image: {filepath}")
    return convert_bgr_to_rgb(image)


class FPSCounter:
    """FPS counter averaged over the last few frames."""

    def __init__(self, buffer_size=30):
        self.timestamps = deque(maxlen=buffer_size)

    def update(self):
        """Record that a frame was shown."""
        self.timestamps.append(time.perf_counter())

    def reset(self):
        self.timestamps.clear()

    def get_fps(self):
        """Calculate current FPS."""
        if len(self.timestamps) < 2:
            return 0.0

        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff == 0:
            return 0.0

        return (len(self.timestamps) - 1) / time_diff
