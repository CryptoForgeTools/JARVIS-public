"""
One-off voice enrollment for speaker verification. Records several
short clips of your voice and averages their embeddings into a single
robust voiceprint (resemblyzer's own recommended approach for this,
via encoder.embed_speaker), saved to disk next to this file.

Speak naturally, varying tone/pace a little across clips -- reading a
sentence normally, then a bit more casually -- for a more robust
profile than one flat reading.

Usage:
    python enroll_voice.py
"""

import os
import numpy as np
import sounddevice as sd
from resemblyzer import VoiceEncoder, preprocess_wav
from config.config import VOICE_PROFILE_PATH

SAMPLE_RATE = 16000
CLIP_SECONDS = 4
NUM_CLIPS = 5
OUTPUT_PATH = str(VOICE_PROFILE_PATH)

encoder = VoiceEncoder(device="cpu")


def record_clip(seconds):
    print(f"Recording {seconds}s... speak naturally.")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def enroll():
    wavs = []
    for i in range(NUM_CLIPS):
        input(f"\nClip {i + 1}/{NUM_CLIPS} -- press Enter, then speak for {CLIP_SECONDS}s...")
        raw = record_clip(CLIP_SECONDS)
        wav = preprocess_wav(raw, source_sr=SAMPLE_RATE)
        if len(wav) == 0:
            print("Got silence/no speech detected in that clip -- redo it.")
            continue
        wavs.append(wav)

    if len(wavs) < 3:
        raise RuntimeError(f"Only got {len(wavs)} usable clips -- need at least 3. Run again.")

    embedding = encoder.embed_speaker(wavs)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, embedding)
    print(f"\nDone -- trained on {len(wavs)} clips.")
    print(f"Saved voice profile to {OUTPUT_PATH}")


if __name__ == "__main__":
    enroll()