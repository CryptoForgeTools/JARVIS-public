"""Central, environment-overridable configuration for JARVIS."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

def _path(key, default):
    value = os.getenv(key, str(default)).strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return (ROOT_DIR / path).resolve() if not path.is_absolute() else path.resolve()

USER_NAME = os.getenv("JARVIS_USER_NAME", "friend").strip() or "friend"
USER_DETAILS = os.getenv("JARVIS_USER_DETAILS", "").strip()
OLLAMA_MODEL = os.getenv("JARVIS_OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TTS_MODEL = os.getenv("JARVIS_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
TTS_SPEAKER = os.getenv("JARVIS_TTS_SPEAKER", "Ryan")
TTS_LANGUAGE = os.getenv("JARVIS_TTS_LANGUAGE", "English")
TTS_REFERENCE_AUDIO = _path("JARVIS_TTS_REFERENCE_AUDIO", "")
TTS_REFERENCE_TEXT = os.getenv("JARVIS_TTS_REFERENCE_TEXT", "")
SHERPA_MODEL_DIR = _path("JARVIS_SHERPA_MODEL_DIR", ROOT_DIR / "models" / "sherpa-onnx-nemotron")
WAKEWORD_MODEL = os.getenv("JARVIS_WAKEWORD_MODEL", "hey_jarvis")
VOICE_PROFILE_PATH = _path("JARVIS_VOICE_PROFILE_PATH", ROOT_DIR / "listeners" / "voice_profile.npy")
FACE_CONFIG_PATH = _path("JARVIS_FACE_CONFIG_PATH", ROOT_DIR / "camera" / "face_model.pkl")
FACE_MODEL_PATH = _path("JARVIS_FACE_MODEL_PATH", ROOT_DIR / "camera" / "face_model.yml")
FACE_DETECTOR_CONFIG = _path("JARVIS_FACE_DETECTOR_CONFIG", ROOT_DIR / "camera" / "deploy.prototxt")
FACE_DETECTOR_MODEL = _path("JARVIS_FACE_DETECTOR_MODEL", ROOT_DIR / "models" / "face" / "res10_300x300_ssd_iter_140000.caffemodel")
MEMORY_PATH = _path("JARVIS_MEMORY_PATH", ROOT_DIR / "memory" / "memory.json")
