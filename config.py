"""
Configuration settings for the face detection system.
"""
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Detection settings
FACE_DETECTION_MODEL = 'hog'  # Options: 'hog' (faster, CPU) or 'cnn' (more accurate, needs a GPU for real time)

# Webcam settings
WEBCAM_INDEX = 0  # Default webcam
PROCESS_EVERY_N_FRAMES = 15  # Run detection on every 15th frame; boxes are reused in between
DETECTION_SCALE = 0.3  # Detect on a frame downscaled to 30% for speed

# Display settings
BBOX_COLOR_FACE = (0, 255, 0)  # Green (BGR)
BBOX_THICKNESS = 2
