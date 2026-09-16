# Real-Time Face Detection

A Python face detection module with a desktop GUI, built as the vision component of a robot project. It finds human faces in a live webcam feed, draws bounding boxes around them and reports the count and frame rate. A command-line tool does the same for photos and whole folders.

It uses the HOG detector from the [`face_recognition`](https://github.com/ageitgey/face_recognition) library (dlib), which runs in real time on a CPU, so it fits small robot computers without a GPU.

## Features

**Webcam GUI** (`webcam_gui.py`)
- Live video with green boxes around every detected face
- Live FPS and face count
- Start / stop the camera, pause / resume detection
- Capture the current annotated frame to `output/`
- Dark resizable Tkinter interface

**Image tool** (`detect_images.py`)
- Detect faces in one image, or batch-process a folder
- Prints each face's coordinates and saves annotated copies

## How real-time detection stays fast

1. **Capture**: frames are read from the webcam with OpenCV.
2. **Downscale**: detection runs on a copy shrunk to 30% of the original size.
3. **Throttle**: detection only runs on every 15th frame; the last boxes are reused in between, so the video stays smooth.
4. **Scale back**: face coordinates are scaled up to the full-size frame before drawing.

All of this happens on Tkinter's main loop (`after()` scheduling), so the interface never gets updated from a background thread.

Settings such as the model, frame interval, scale and camera index live in [`config.py`](config.py).

| Model | Speed | Accuracy | Hardware |
|---|---|---|---|
| `hog` (default) | Real time on CPU | Good for frontal faces | Any CPU |
| `cnn` | Slow on CPU | Better with angles and lighting | GPU recommended |

## Installation

Requires **Python 3.9–3.12** and a webcam.

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

> **dlib on Windows**: `face-recognition` depends on `dlib`, which pip compiles from source and needs CMake and the Visual Studio C++ build tools. If the install fails, install a prebuilt dlib wheel for your Python version first, then run `pip install -r requirements.txt` again.

## Usage

**Webcam GUI**

```bash
python webcam_gui.py
```

Click **Start Camera**. Captured frames are saved in `output/`.

**Images**

```bash
# One image, shown in a window
python detect_images.py --input photo.jpg --show

# A whole folder, annotated copies saved to results/
python detect_images.py --input photos/ --output results/
```

## Using it in another project

The detector is a small class that can be imported from any control script, for example a robot's main loop:

```python
import cv2
from face_detector import FaceDetector

detector = FaceDetector()          # or FaceDetector(model="cnn")
camera = cv2.VideoCapture(0)

ok, frame = camera.read()
faces = detector.detect_faces_in_frame(frame, scale=0.3)
# [(top, right, bottom, left), ...] in full-frame pixel coordinates
```

## Project structure

```
├── webcam_gui.py      # Desktop app: live webcam detection
├── detect_images.py   # Command-line detection for images and folders
├── face_detector.py   # FaceDetector class (wraps face_recognition)
├── utils.py           # Image conversion, drawing, saving, FPS counter
├── config.py          # Model, performance and display settings
└── requirements.txt
```

## Limitations

- This is face **detection** (where faces are), not identification (who they are).
- The HOG model works best on faces looking roughly toward the camera, and can miss faces that are very small, turned away or in poor lighting.
