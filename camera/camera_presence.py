"""
Camera presence detection for JARVIS.

Runs on CPU, at low frequency, independent of the GPU-bound LLM/TTS
pipeline. Two stages:

  1. FaceDetector — OpenCV's DNN SSD face detector (res10_300x300).
     Fast, no VRAM, robust to lighting/angle. Needs two small model
     files you download once (see DOWNLOAD NOTE below).

  2. FaceRecognizer — OpenCV's LBPH recognizer (opencv-contrib-python).
     Chosen over the more common `face_recognition`/dlib specifically
     to avoid a CMake + Visual Studio Build Tools compile on Windows.
     You enroll a handful of your own face crops once; after that it
     scores new detections against your enrolled identity.

PresenceMonitor ties both together with a debounce + cooldown state
machine (same shape as the clap detector's double-clap gate): a face
has to be present across several consecutive checks before it counts
as an "arrival," and a cooldown prevents re-greeting every few seconds
while you're just sitting there.

DEPENDENCY WARNING: installing opencv-python/opencv-contrib-python can
silently move your pinned numpy version. Run `pip show numpy` right
after installing and re-pin to 1.26.4 if it shifted — same failure
mode you've already hit with onnxruntime/openwakeword.

DOWNLOAD NOTE: FaceDetector needs two files from OpenCV's official repo
(not included here — small binary/model files):
  - deploy.prototxt
  - res10_300x300_ssd_iter_140000.caffemodel
Both are in opencv/opencv (extra/testdata/dnn) and opencv/opencv_3rdparty
on GitHub — search "res10_300x300_ssd_iter_140000" if the paths move.
Place them next to this file or pass explicit paths to FaceDetector().
"""

import os
import time
import cv2
import numpy as np
from config.config import FACE_DETECTOR_CONFIG, FACE_DETECTOR_MODEL

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


class FaceDetector:
    def __init__(
        self,
        prototxt_path: str = None,
        model_path: str = None,
        confidence_threshold: float = 0.6,
    ):
        # default to files sitting next to THIS file, regardless of the
        # caller's current working directory
        if prototxt_path is None:
            prototxt_path = str(FACE_DETECTOR_CONFIG)
        if model_path is None:
            model_path = str(FACE_DETECTOR_MODEL)
        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                "Face detector model files missing — see DOWNLOAD NOTE at the "
                "top of this file."
            )
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame_bgr: np.ndarray):
        """Returns a list of (x1, y1, x2, y2, confidence) boxes in frame coords."""
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            boxes.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2), confidence))
        return boxes


class FaceRecognizer:
    """LBPH recognizer — enroll a handful of your own face crops, then
    score new detections against that single identity."""

    def __init__(self, distance_threshold: float = 70.0):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.distance_threshold = distance_threshold  # lower = more confident match
        self._trained = False

    def enroll(self, face_crops_gray: list):
        """
        face_crops_gray: list of grayscale face-crop images (numpy arrays,
        same-ish size, e.g. 200x200) of YOU, ideally from a few sessions —
        different lighting/angle/glasses-on-off improves robustness.
        """
        labels = np.zeros(len(face_crops_gray), dtype=np.int32)  # single identity: label 0
        self.recognizer.train(face_crops_gray, labels)
        self._trained = True

    def is_me(self, face_crop_gray: np.ndarray) -> bool:
        if not self._trained:
            raise RuntimeError("Call enroll() before is_me().")
        label, distance = self.recognizer.predict(face_crop_gray)
        return label == 0 and distance < self.distance_threshold


class PresenceMonitor:
    def __init__(
        self,
        detector: FaceDetector,
        recognizer: FaceRecognizer = None,   # None = greet ANY face, not just you
        camera_index: int = 0,
        check_interval_s: float = 0.75,      # how often to grab+check a frame
        confirm_frames: int = 3,             # consecutive present-checks before "arrived"
        cooldown_s: float = 300.0,           # min gap between greetings
        on_arrival=None,                     # callback fired on confirmed arrival
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.camera_index = camera_index
        self.check_interval_s = check_interval_s
        self.confirm_frames = confirm_frames
        self.cooldown_s = cooldown_s
        self.on_arrival = on_arrival

        self._present_streak = 0
        self._currently_marked_present = False
        self._last_greeting = 0.0
        self._running = False

    def run_forever(self):
        """Blocking loop — run this in its own thread from main.py."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(
                f"[PresenceMonitor] ERROR: could not open camera index "
                f"{self.camera_index} — is another process (a leftover "
                f"test_presence.py / diagnose_presence.py window) still "
                f"holding it?"
            )
            return

        self._running = True
        consecutive_failures = 0
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    consecutive_failures += 1
                    if consecutive_failures == 30:
                        print(
                            f"[PresenceMonitor] WARNING: {consecutive_failures} "
                            f"consecutive frame-grab failures — camera may be "
                            f"busy or disconnected."
                        )
                    time.sleep(self.check_interval_s)
                    continue
                consecutive_failures = 0

                present = self._check_frame(frame)

                if present:
                    self._present_streak += 1
                else:
                    self._present_streak = 0
                    self._currently_marked_present = False

                if (
                    present
                    and not self._currently_marked_present
                    and self._present_streak >= self.confirm_frames
                ):
                    now = time.monotonic()
                    if now - self._last_greeting >= self.cooldown_s:
                        self._currently_marked_present = True
                        self._last_greeting = now
                        if self.on_arrival:
                            self.on_arrival()

                time.sleep(self.check_interval_s)
        finally:
            cap.release()

    def stop(self):
        self._running = False

    def _check_frame(self, frame_bgr: np.ndarray) -> bool:
        boxes = self.detector.detect(frame_bgr)
        if not boxes:
            return False

        if self.recognizer is None:
            return True  # any face counts

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        for (x1, y1, x2, y2, _conf) in boxes:
            crop = gray[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (200, 200))
            if self.recognizer.is_me(crop):
                return True
        return False


# --- wiring sketch for main.py ---------------------------------------------
#
# detector = FaceDetector()
# recognizer = FaceRecognizer()
# recognizer.enroll(your_collected_face_crops)   # do this once, save/reload as needed
#
# def greet():
#     hour = time.localtime().tm_hour
#     greeting = "Good morning" if hour < 12 else "Good evening" if hour >= 18 else "Hey"
#     speak(f"{greeting}")     # connect your configured TTS call
#     start_listening_session()        # optional: open mic like a wake-word trigger
#
# monitor = PresenceMonitor(detector, recognizer, on_arrival=greet)
# threading.Thread(target=monitor.run_forever, daemon=True).start()