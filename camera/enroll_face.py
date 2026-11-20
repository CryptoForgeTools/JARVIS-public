"""
One-off enrollment script for camera_presence.py's FaceRecognizer.

Opens the laptop camera, shows a live preview, and lets you capture
face crops with SPACE. Move around between captures — different
angles, lighting, glasses on/off — to make the recognizer robust.
Press ESC when done (aim for ~15-20 captures).

Trains the LBPH recognizer on what you captured and pickles it to
disk, so main.py can load it once at startup instead of retraining
every boot.

Usage:
    python enroll_face.py
    then run main.py after enrollment is complete
"""

import os
import pickle
import cv2
from config.config import FACE_CONFIG_PATH, FACE_MODEL_PATH

try:
    # works when imported as part of the camera package, e.g. main.py's
    # `from camera.enroll_face import load_recognizer`
    from camera.camera_presence import FaceDetector, FaceRecognizer
except ImportError:
    # works when run directly as a standalone script from inside camera/,
    # e.g. `python enroll_face.py` for re-enrolling later
    from camera_presence import FaceDetector, FaceRecognizer

CAMERA_INDEX = 0
# absolute, so it always saves/loads next to THIS file regardless of
# whatever directory you happen to run the script from
OUTPUT_PATH = str(FACE_CONFIG_PATH)
FACE_MODEL_OUTPUT_PATH = str(FACE_MODEL_PATH)
TARGET_CAPTURES = 20


def enroll():
    detector = FaceDetector()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    crops = []
    print("SPACE = capture this frame   |   ESC = finish and train")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        boxes = detector.detect(frame)
        display = frame.copy()
        for (x1, y1, x2, y2, conf) in boxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display, f"{conf:.2f}", (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        cv2.putText(
            display, f"Captured: {len(crops)}/{TARGET_CAPTURES}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )
        cv2.imshow("Enroll face - SPACE to capture, ESC to finish", display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key == 32:  # SPACE
            if not boxes:
                print("No face detected — try again.")
                continue
            # use the largest/most confident box if multiple faces are in frame
            x1, y1, x2, y2, _conf = max(boxes, key=lambda b: b[4])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop = gray[y1:y2, x1:x2]
            if crop.size == 0:
                print("Bad crop — try again.")
                continue
            crop = cv2.resize(crop, (200, 200))
            crops.append(crop)
            print(f"Captured {len(crops)}/{TARGET_CAPTURES}")

    cap.release()
    cv2.destroyAllWindows()

    if len(crops) < 5:
        raise RuntimeError(
            f"Only got {len(crops)} captures — need at least 5 for a usable "
            "model. Run again and capture more."
        )

    recognizer = FaceRecognizer()
    recognizer.enroll(crops)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FACE_MODEL_OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        # LBPHFaceRecognizer itself isn't directly picklable in older
        # OpenCV builds, so save it via its own .write() and wrap the
        # rest of the config alongside it.
        recognizer.recognizer.write(FACE_MODEL_OUTPUT_PATH)
        pickle.dump(
            {"distance_threshold": recognizer.distance_threshold},
            f,
        )

    print(f"Done — trained on {len(crops)} captures.")
    print(f"Saved face model to {FACE_MODEL_OUTPUT_PATH} and config to {OUTPUT_PATH}")


def load_recognizer(config_path: str = OUTPUT_PATH) -> FaceRecognizer:
    """Load an enrolled recognizer; return None until enrollment is complete."""
    if not os.path.exists(config_path) or not os.path.exists(FACE_MODEL_OUTPUT_PATH):
        return None
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    recognizer = FaceRecognizer(distance_threshold=config["distance_threshold"])
    recognizer.recognizer.read(FACE_MODEL_OUTPUT_PATH)
    recognizer._trained = True
    return recognizer


if __name__ == "__main__":
    enroll()