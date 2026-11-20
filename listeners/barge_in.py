"""
Barge-in listener for JARVIS.

Runs *only while AudioManager is playing TTS audio*. Watches live mic
energy and, if it sees speech-shaped energy clearly above JARVIS's own
expected self-bleed level, interrupts playback and fires a callback so
main.py can drop straight into listening mode for the user's new
utterance.

IMPORTANT — this is a calibrated energy gate, not real acoustic echo
cancellation:
  - On headphones, self-bleed is ~0, so this is close to bulletproof —
    almost any mic energy above room noise is real speech.
  - On open speakers, self-bleed scales with volume and speaker/mic
    placement, so calibrate_self_bleed() needs to run whenever the
    volume or physical setup changes, and leak_margin may need tuning.

Integration points in main.py:
  - Call monitor.arm() the instant TTS playback starts.
  - Call monitor.disarm() when playback ends normally.
  - In on_barge_in: cancel/drop the in-flight generation + TTS queues,
    SKIP the existing post-playback wake-word cooldown (this interrupt
    is intentional, not JARVIS hearing itself), and route straight into
    your existing sherpa-onnx streaming capture for the new utterance.
"""

import numpy as np
import sounddevice as sd


class BargeInMonitor:
    def __init__(
        self,
        audio_manager,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        leak_margin: float = 1.6,   # how far above measured self-bleed = "real speech"
        sustain_ms: int = 180,      # must stay above threshold this long (ignores clicks/pops)
        min_threshold: float = 0.01,  # floor so a silent/near-zero calibration can't over-trigger
        on_barge_in=None,
    ):
        self.audio_manager = audio_manager
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.leak_margin = leak_margin
        self.sustain_frames = max(1, sustain_ms // frame_ms)
        self.min_threshold = min_threshold
        self.on_barge_in = on_barge_in

        self.self_bleed_rms = 0.0
        self._stream = None
        self._active = False
        self._above_count = 0

    def calibrate_self_bleed(self, playback_audio: np.ndarray, playback_samplerate: int):
        """
        Run once at startup (and again if output volume or speaker/mic
        placement changes): plays a representative TTS clip through
        AudioManager while recording the mic, to measure how loud
        JARVIS's own voice looks from the mic's point of view.
        """
        duration = len(playback_audio) / playback_samplerate
        rec = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        self.audio_manager.play(playback_audio, playback_samplerate)
        sd.wait()
        self.self_bleed_rms = float(np.sqrt(np.mean(rec.astype(np.float64) ** 2)))

    def arm(self):
        """Start monitoring — call the instant TTS playback begins."""
        self._active = True
        self._above_count = 0
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_size,
            callback=self._callback,
        )
        self._stream.start()

    def disarm(self):
        """Stop monitoring — call when playback ends normally."""
        self._active = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if not self._active:
            return

        rms = float(np.sqrt(np.mean(indata[:, 0].astype(np.float64) ** 2)))
        threshold = max(self.self_bleed_rms * self.leak_margin, self.min_threshold)

        if rms > threshold:
            self._above_count += 1
        else:
            self._above_count = 0

        if self._above_count >= self.sustain_frames:
            self._active = False
            self.audio_manager.interrupt()
            if self.on_barge_in:
                self.on_barge_in()