"""
Face detector module for finding faces in images.
"""
import cv2
import numpy as np
import face_recognition
from config import FACE_DETECTION_MODEL


class FaceDetector:
    """Face detection using the face_recognition library."""

    def __init__(self, model=FACE_DETECTION_MODEL):
        """
        Initialize the face detector.

        Args:
            model: Detection model ('hog' or 'cnn')
        """
        self.model = model

    def detect_faces(self, image):
        """
        Detect faces in an image.

        Args:
            image: RGB image as numpy array

        Returns:
            List of face locations (top, right, bottom, left)
        """
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        return face_recognition.face_locations(image, model=self.model)

    def detect_faces_in_frame(self, bgr_frame, scale=1.0):
        """
        Detect faces in an OpenCV (BGR) frame, optionally on a downscaled copy for speed.

        Args:
            bgr_frame: Frame as returned by cv2.VideoCapture.read()
            scale: Factor to shrink the frame by before detection (e.g. 0.3)

        Returns:
            List of face locations (top, right, bottom, left) in the original frame's coordinates
        """
        small = bgr_frame if scale == 1.0 else cv2.resize(bgr_frame, (0, 0), fx=scale, fy=scale)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        return [
            (int(top / scale), int(right / scale), int(bottom / scale), int(left / scale))
            for (top, right, bottom, left) in self.detect_faces(rgb)
        ]
