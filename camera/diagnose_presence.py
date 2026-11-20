"""
Live diagnostic for camera presence — prints what's happening on EVERY
frame, not just on confirmed arrival, so you can tell whether the
problem is detection (no face found), recognition (face found but not
matched as you), or the camera itself (frame grabs failing, usually a
sign something else still has it open exclusively).

Ctrl+C to stop.
"""

import os
import cv2
from camera_presence import FaceDetector
from enroll_face import load_recognizer
from config.config import FACE_CONFIG_PATH

CAMERA_INDEX = 0
_MODEL_PATH = str(FACE_CONFIG_PATH)

detector = FaceDetector()
recognizer = load_recognizer(_MODEL_PATH)
if recognizer is None:
    raise SystemExit("No enrolled face model found. Run python camera/enroll_face.py first.")

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Camera did not open at all — is another process (another "
          "terminal still running test_presence.py, Zoom, the Windows "
          "Camera app) already using it?")
    raise SystemExit(1)

print("Watching... Ctrl+C to stop.\n")
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab FAILED — camera busy/exclusive-locked by another process?")
            continue

        boxes = detector.detect(frame)
        if not boxes:
            print("No face detected in this frame.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x1, y1, x2, y2, conf) in boxes:
            crop = gray[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (200, 200))
            label, distance = recognizer.recognizer.predict(crop)
            match = "MATCH" if distance < recognizer.distance_threshold else "no match"
            print(
                f"Face found (detector conf {conf:.2f}) -> recognizer distance "
                f"{distance:.1f} (threshold {recognizer.distance_threshold}) -> {match}"
            )
except KeyboardInterrupt:
    pass
finally:
    cap.release()