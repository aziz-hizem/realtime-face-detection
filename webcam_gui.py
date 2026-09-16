"""
Modern GUI for Real-time Webcam Face Detection
Professional interface with smooth animations and controls
"""
import os
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk

from config import (
    WEBCAM_INDEX, PROCESS_EVERY_N_FRAMES, DETECTION_SCALE,
    BBOX_COLOR_FACE, BBOX_THICKNESS, OUTPUT_DIR
)
from face_detector import FaceDetector
from utils import FPSCounter, convert_bgr_to_rgb, draw_face_box, save_image

FRAME_DELAY_MS = 10  # Pause between frame updates, keeps the UI responsive


class ModernFaceRecognitionGUI:
    """Modern GUI for face detection system"""

    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x750")
        self.root.resizable(True, True)

        # Modern color scheme
        self.colors = {
            'bg': '#1e1e2e',
            'panel': '#2d2d44',
            'accent': '#7c3aed',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'text': '#e5e7eb',
            'text_secondary': '#9ca3af'
        }

        self.root.configure(bg=self.colors['bg'])

        # Initialize variables
        self.detector = FaceDetector()
        self.video_capture = None
        self.is_running = False
        self.detection_enabled = True
        self.frame_count = 0
        self.fps_counter = FPSCounter()
        self.face_results = []
        self.current_frame = None

        # Build UI
        self.setup_styles()
        self.create_widgets()

        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        """Configure ttk styles for modern look"""
        style = ttk.Style()
        style.theme_use('clam')

        # Button styles
        style.configure('Accent.TButton',
                       background=self.colors['accent'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=10,
                       font=('Segoe UI', 10, 'bold'))

        style.map('Accent.TButton',
                 background=[('active', '#6d28d9')])

        style.configure('Success.TButton',
                       background=self.colors['success'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=10,
                       font=('Segoe UI', 10, 'bold'))

        style.map('Success.TButton',
                 background=[('active', '#059669')])

        style.configure('Danger.TButton',
                       background=self.colors['danger'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=10,
                       font=('Segoe UI', 10, 'bold'))

        style.map('Danger.TButton',
                 background=[('active', '#dc2626')])

    def create_widgets(self):
        """Create all UI widgets"""

        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['panel'], height=80)
        header_frame.pack(fill='x', padx=0, pady=0)
        header_frame.pack_propagate(False)

        title_label = tk.Label(header_frame,
                      text="🎯 Face Detection System",
                              font=('Segoe UI', 24, 'bold'),
                              bg=self.colors['panel'],
                              fg=self.colors['text'])
        title_label.pack(pady=20)

        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill='both', expand=True, padx=20, pady=20)

        # Left panel - Video feed
        left_panel = tk.Frame(main_container, bg=self.colors['panel'], relief='flat', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Video canvas
        self.video_canvas = tk.Canvas(left_panel, bg='black', highlightthickness=0)
        self.video_canvas.pack(fill='both', expand=True, padx=10, pady=10)

        # Right panel - Controls and stats
        right_panel = tk.Frame(main_container, bg=self.colors['panel'], width=300)
        right_panel.pack(side='right', fill='y')
        right_panel.pack_propagate(False)

        # Status section
        status_frame = tk.LabelFrame(right_panel, text="Status",
                                    bg=self.colors['panel'],
                                    fg=self.colors['text'],
                                    font=('Segoe UI', 12, 'bold'),
                                    relief='flat')
        status_frame.pack(fill='x', padx=10, pady=10)

        self.status_label = tk.Label(status_frame,
                                     text="⚪ Ready",
                                     font=('Segoe UI', 11),
                                     bg=self.colors['panel'],
                                     fg=self.colors['text_secondary'],
                                     anchor='w')
        self.status_label.pack(fill='x', padx=10, pady=5)

        # Stats section
        stats_frame = tk.LabelFrame(right_panel, text="Statistics",
                                   bg=self.colors['panel'],
                                   fg=self.colors['text'],
                                   font=('Segoe UI', 12, 'bold'),
                                   relief='flat')
        stats_frame.pack(fill='x', padx=10, pady=10)

        self.fps_label = tk.Label(stats_frame,
                                 text="FPS: 0",
                                 font=('Segoe UI', 11),
                                 bg=self.colors['panel'],
                                 fg=self.colors['success'],
                                 anchor='w')
        self.fps_label.pack(fill='x', padx=10, pady=5)

        self.faces_label = tk.Label(stats_frame,
                                   text="Faces Detected: 0",
                                   font=('Segoe UI', 11),
                                   bg=self.colors['panel'],
                                   fg=self.colors['text_secondary'],
                                   anchor='w')
        self.faces_label.pack(fill='x', padx=10, pady=2)

        # Controls section
        controls_frame = tk.Frame(right_panel, bg=self.colors['panel'])
        controls_frame.pack(fill='x', padx=10, pady=10)

        self.start_btn = ttk.Button(controls_frame,
                                    text="▶ Start Camera",
                                    style='Success.TButton',
                                    command=self.toggle_camera)
        self.start_btn.pack(fill='x', pady=5)

        self.detect_btn = ttk.Button(controls_frame,
                                     text="⏸ Pause Detection",
                                     style='Accent.TButton',
                                     command=self.toggle_detection,
                                     state='disabled')
        self.detect_btn.pack(fill='x', pady=5)

        self.capture_btn = ttk.Button(controls_frame,
                                      text="📸 Capture Frame",
                                      style='Accent.TButton',
                                      command=self.capture_frame,
                                      state='disabled')
        self.capture_btn.pack(fill='x', pady=5)

        ttk.Button(controls_frame,
                  text="❌ Exit",
                  style='Danger.TButton',
                  command=self.on_closing).pack(fill='x', pady=5)

    def toggle_camera(self):
        """Start or stop the camera"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Initialize and start the webcam"""
        try:
            self.video_capture = cv2.VideoCapture(WEBCAM_INDEX)

            if not self.video_capture.isOpened():
                messagebox.showerror("Error", "Could not open webcam!\nPlease check your camera.")
                return

            self.is_running = True
            self.frame_count = 0
            self.face_results = []
            self.fps_counter.reset()
            self.start_btn.config(text="⏹ Stop Camera")
            self.detect_btn.config(state='normal')
            self.capture_btn.config(state='normal')
            self.update_status()

            # Frames are read and drawn on Tkinter's own thread, the only one allowed to touch widgets
            self.root.after(0, self.update_frame)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera:\n{str(e)}")

    def stop_camera(self):
        """Stop the webcam"""
        self.is_running = False

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        self.start_btn.config(text="▶ Start Camera")
        self.detect_btn.config(state='disabled')
        self.capture_btn.config(state='disabled')
        self.status_label.config(text="⚪ Stopped", fg=self.colors['text_secondary'])

        # Clear canvas and stats
        self.video_canvas.delete("all")
        self.face_results = []
        self.fps_label.config(text="FPS: 0")
        self.faces_label.config(text="Faces Detected: 0")

    def toggle_detection(self):
        """Toggle face detection on/off"""
        self.detection_enabled = not self.detection_enabled

        if not self.detection_enabled:
            self.face_results = []  # Don't leave stale boxes on screen while paused
        self.detect_btn.config(text="⏸ Pause Detection" if self.detection_enabled else "▶ Resume Detection")
        self.update_status()

    def update_status(self):
        if self.detection_enabled:
            self.status_label.config(text="🟢 Live - detecting faces", fg=self.colors['success'])
        else:
            self.status_label.config(text="⏸ Live - detection paused", fg=self.colors['warning'])

    def update_frame(self):
        """Read, process and show one frame, then schedule the next one"""
        if not self.is_running:
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.stop_camera()
            messagebox.showerror("Error", "Lost the webcam feed.")
            return

        self.frame_count += 1
        self.fps_counter.update()

        # Detection is the slow part, so it only runs on every Nth frame
        if self.detection_enabled and self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            self.face_results = self.detector.detect_faces_in_frame(frame, DETECTION_SCALE)

        display_frame = self.draw_results(frame)
        self.current_frame = display_frame
        self.display_frame(display_frame)
        self.update_stats()

        self.root.after(FRAME_DELAY_MS, self.update_frame)

    def draw_results(self, frame):
        """Draw bounding boxes on a copy of the frame"""
        display_frame = frame.copy()
        for location in self.face_results:
            draw_face_box(display_frame, location, BBOX_COLOR_FACE, BBOX_THICKNESS)
        return display_frame

    def display_frame(self, frame):
        """Display frame on canvas"""
        frame_rgb = convert_bgr_to_rgb(frame)

        # Get canvas size
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            # Resize frame to fit canvas while maintaining aspect ratio
            frame_height, frame_width = frame_rgb.shape[:2]
            scale = min(canvas_width / frame_width, canvas_height / frame_height)
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)

            resized_frame = cv2.resize(frame_rgb, (new_width, new_height))

            # Convert to PhotoImage
            img = Image.fromarray(resized_frame)
            photo = ImageTk.PhotoImage(image=img)

            # Update canvas
            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=photo,
                anchor='center'
            )
            self.video_canvas.image = photo  # Keep reference

    def update_stats(self):
        """Update statistics display"""
        fps = self.fps_counter.get_fps()
        self.fps_label.config(text=f"FPS: {fps:.1f}")

        total_faces = len(self.face_results)
        self.faces_label.config(text=f"Faces Detected: {total_faces}")

    def capture_frame(self):
        """Save current frame to file"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame to capture!")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            filepath = save_image(self.current_frame, os.path.join(OUTPUT_DIR, filename))

            messagebox.showinfo("Success", f"Frame saved!\n\n{filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save frame:\n{str(e)}")

    def on_closing(self):
        """Handle window close event"""
        if self.is_running:
            self.stop_camera()

        self.root.quit()
        self.root.destroy()


def main():
    """Launch the GUI application"""
    root = tk.Tk()
    ModernFaceRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
