"""
Clap-to-wake detector for JARVIS.

openWakeWord is a keyword classifier and won't help here — this is a
separate, lightweight energy-transient detector tuned to the acoustic
signature of a hand clap: a fast attack, broadband burst, that decays
within tens of milliseconds (unlike sustained noise like speech, music,
or a door closing). Requires a *double* clap within a short window
before firing, same pattern as old clap-activated light switches, to
cut down on false wakes from random bangs.

Designed to push into the SAME wake-event queue/callback your
hey_jarvis detection already uses, so either one starts a session.

Integration note: on Windows, running this as its own InputStream
alongside openWakeWord's own stream can fight over the mic device
(WASAPI exclusive-mode issues). Prefer opening ONE shared InputStream
in main.py and feeding frames to both detectors, rather than each
module owning its own stream — see the bottom of this file for the
shared-stream call shape.
"""

import time
import numpy as np
import sounddevice as sd
from collections import deque


class ClapDetector:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 10,
        onset_ratio: float = 6.0,        # peak energy vs rolling floor to count as a transient
        min_peak_rms: float = 0.02,      # absolute floor so a silent room can't self-trigger
        max_clap_duration_ms: int = 60,  # claps decay fast; sustained noise doesn't
        clap_window_s: float = 0.8,      # time window to catch the *second* clap
        cooldown_s: float = 1.5,         # ignore new claps right after a trigger
        on_wake=None,
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.onset_ratio = onset_ratio
        self.min_peak_rms = min_peak_rms
        self.max_clap_frames = max(1, max_clap_duration_ms // frame_ms)
        self.clap_window_s = clap_window_s
        self.cooldown_s = cooldown_s
        self.on_wake = on_wake

        self._floor = 1e-4          # rolling noise-floor estimate
        self._floor_alpha = 0.02    # smoothing factor for the floor
        self._recent_claps = deque()
        self._last_trigger = 0.0
        self._in_transient = False
        self._transient_frames = 0

        self._stream = None
        self._running = False

    # --- public control ----------------------------------------------

    def start(self):
        """Opens its own InputStream. Skip this if using a shared stream
        (see process_frame below) and just call process_frame() yourself."""
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_size,
            callback=lambda indata, frames, t, status: self.process_frame(indata[:, 0]),
        )
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def calibrate(self, seconds: float = 2.0):
        """Run at startup in a quiet room to seed the noise floor."""
        rec = sd.rec(
            int(seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        rms = float(np.sqrt(np.mean(rec.astype(np.float64) ** 2)))
        self._floor = max(rms, 1e-5)

    def process_frame(self, frame: np.ndarray):
        """
        Feed one frame (frame_size samples, mono float32) at a time.
        Call this directly if you're running a shared mic InputStream
        instead of letting this class own its own stream.
        """
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        now = time.monotonic()

        is_transient = rms > self.min_peak_rms and rms > self._floor * self.onset_ratio

        if is_transient:
            self._transient_frames += 1
            self._in_transient = True
        else:
            if self._in_transient:
                if self._transient_frames <= self.max_clap_frames:
                    self._register_clap(now)
                self._transient_frames = 0
                self._in_transient = False
            # only adapt the floor during quiet frames, so a clap or
            # loud playback never teaches the detector to expect it
            self._floor = (1 - self._floor_alpha) * self._floor + self._floor_alpha * rms

    # --- internals ------------------------------------------------------

    def _register_clap(self, now: float):
        while self._recent_claps and now - self._recent_claps[0] > self.clap_window_s:
            self._recent_claps.popleft()
        self._recent_claps.append(now)

        if now - self._last_trigger < self.cooldown_s:
            return

        if len(self._recent_claps) >= 2:
            self._last_trigger = now
            self._recent_claps.clear()
            if self.on_wake:
                self.on_wake()


# --- shared-stream sketch, for wiring into main.py alongside openWakeWord ---
#
# clap = ClapDetector(on_wake=lambda: wake_queue.put("clap"))
# clap.calibrate()
#
# def shared_mic_callback(indata, frames, time_info, status):
#     mono = indata[:, 0]
#     openwakeword_model.predict(mono)        # your existing hey_jarvis path
#     clap.process_frame(mono)                # feed the same frame here too
#
# with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
#                      blocksize=clap.frame_size, callback=shared_mic_callback):
#     ...