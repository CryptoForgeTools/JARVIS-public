"""
Standalone smoke test for camera presence detection.

Prints to console instead of speaking, so you can confirm the whole
detect -> recognize -> debounce -> callback chain works before wiring
it into main.py's TTS. Walk toward/away from the camera a few times
while this runs.

Ctrl+C to stop.
"""

import time
from camera_presence import FaceDetector, PresenceMonitor
from enroll_face import load_recognizer
from config.config import FACE_CONFIG_PATH, USER_NAME

detector = FaceDetector()
recognizer = load_recognizer(str(FACE_CONFIG_PATH))
if recognizer is None:
    raise SystemExit("No enrolled face model found. Run python camera/enroll_face.py first.")


def on_arrival():
    hour = time.localtime().tm_hour
    greeting = "Good morning" if hour < 12 else "Good evening" if hour >= 18 else "Hey"
    print(f"[PRESENCE] Arrival confirmed -> would say: '{greeting}, {USER_NAME}.'")


monitor = PresenceMonitor(
    detector,
    recognizer,
    on_arrival=on_arrival,
)

print("Watching for you... walk out of frame and back in to trigger a re-greeting.")
print("(cooldown is 300s by default, so re-greets are intentionally slow)")

try:
    monitor.run_forever()
except KeyboardInterrupt:
    monitor.stop()
    print("Stopped.")