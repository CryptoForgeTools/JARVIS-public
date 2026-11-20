"""
Speaker verification for JARVIS -- distinguishes YOUR voice from anyone
else's talking in the room, using resemblyzer (a lightweight ~256-dim
speaker embedding model). Deliberately pinned to CPU: your GPU budget
is already fully committed to Ollama + TTS, and this doesn't need to
be fast enough to compete for it.

Enrollment: run enroll_voice.py once to create voice_profile.npy.
Runtime: is_me(audio) embeds a captured utterance and compares it to
the enrolled profile via cosine similarity.

Fails OPEN (returns True) if there's no profile yet, no usable audio,
or the embedding call itself errors -- same philosophy as
Brain.is_addressed_to_me(): better to occasionally let a stranger's
utterance through than to randomly go deaf to your own real commands.
"""

import os
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from config.config import VOICE_PROFILE_PATH

_PROFILE_PATH = str(VOICE_PROFILE_PATH)


class SpeakerVerifier:
    def __init__(self, similarity_threshold: float = 0.75, sample_rate: int = 16000):
        self.encoder = None
        self.similarity_threshold = similarity_threshold
        self.sample_rate = sample_rate
        self.profile = None
        if os.path.exists(_PROFILE_PATH):
            self.profile = np.load(_PROFILE_PATH)
            self.encoder = VoiceEncoder(device="cpu")
        else:
            print(
                "[speaker_id] no enrolled voice profile found -- run "
                "enroll_voice.py first, or everyone is treated as you for now."
            )

    @property
    def enrolled(self) -> bool:
        return self.profile is not None

    def is_me(self, audio: np.ndarray) -> bool:
        if self.profile is None or audio is None or len(audio) == 0:
            return True
        try:
            wav = preprocess_wav(audio, source_sr=self.sample_rate)
            if len(wav) == 0:
                return True
            if self.encoder is None:
                self.encoder = VoiceEncoder(device="cpu")
            embedding = self.encoder.embed_utterance(wav)
            similarity = float(np.dot(embedding, self.profile))
            return similarity >= self.similarity_threshold
        except Exception as e:
            print(f"[speaker_id] verification failed, defaulting to YES: {e}")
            return True