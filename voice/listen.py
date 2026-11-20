# voice/listen.py
import sherpa_onnx
import sounddevice as sd
import numpy as np
from config.config import SHERPA_MODEL_DIR

class Listener:
    def __init__(self, model_dir=None,
                 sample_rate=16000, num_threads=4):
        model_dir = str(model_dir or SHERPA_MODEL_DIR)
        self.sample_rate = sample_rate
        self.window_size = 960  # 60ms chunks

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{model_dir}/tokens.txt",
            encoder=f"{model_dir}/encoder.onnx",
            decoder=f"{model_dir}/decoder.onnx",
            joiner=f"{model_dir}/joiner.onnx",
            num_threads=num_threads,
            sample_rate=sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=0.7,
            rule3_min_utterance_length=20.0,
        )
        self.stream = self.recognizer.create_stream()

    def listen_and_transcribe(self, max_seconds=15):
        """Returns (transcribed_text, raw_audio) -- raw_audio is the full
        captured mono float32 buffer at self.sample_rate, needed for
        speaker verification, not just the transcript."""
        print("Listening...")
        self.stream = self.recognizer.create_stream()
        max_chunks = int(max_seconds * self.sample_rate / self.window_size)
        audio_chunks = []

        with sd.InputStream(channels=1, samplerate=self.sample_rate,
                             dtype='float32', blocksize=self.window_size) as mic:
            last_partial = ""
            for _ in range(max_chunks):
                audio, _ = mic.read(self.window_size)
                audio = np.asarray(audio[:, 0], dtype=np.float32)
                audio_chunks.append(audio.copy())

                self.stream.accept_waveform(self.sample_rate, audio)
                while self.recognizer.is_ready(self.stream):
                    self.recognizer.decode_stream(self.stream)

                partial = self.recognizer.get_result(self.stream).strip()
                if partial and partial != last_partial:
                    # \r overwrites the same line so it reads as live-updating
                    # text, not a scroll of every intermediate guess
                    print(f"\r> {partial}" + " " * 10, end="", flush=True)
                    last_partial = partial

                if self.recognizer.is_endpoint(self.stream):
                    print()  # move off the live-update line once the utterance is done
                    full_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)
                    return partial, full_audio

        print()
        text = self.recognizer.get_result(self.stream).strip()
        full_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)
        return text, full_audio