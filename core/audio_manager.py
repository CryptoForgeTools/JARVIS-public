import threading
import time
import sounddevice as sd

class AudioManager:
    """The ONLY component allowed to touch the speaker. Serializes all playback
    and gates mic listening on whether JARVIS is currently speaking."""
    def __init__(self, post_speech_settle=0.4, interrupt_chunk_ms=20):
        self._lock = threading.Lock()
        self._speaking = threading.Event()
        self.post_speech_settle = post_speech_settle
        self.interrupt_chunk_ms = interrupt_chunk_ms
        # set by interrupt() (e.g. from a barge-in listener) to cut the
        # CURRENT play()/play_stream() call short; cleared at the start
        # of each new call
        self.interrupt_event = threading.Event()

    @property
    def is_speaking(self):
        return self._speaking.is_set()

    def interrupt(self):
        """Cuts whatever is currently playing short, as soon as the playback
        loop notices (within one interrupt_chunk_ms window)."""
        self.interrupt_event.set()

    def play(self, audio_array, sample_rate):
        """Returns True if playback completed normally, False if interrupted."""
        if audio_array is None or len(audio_array) == 0:
            return True
        with self._lock:
            self.interrupt_event.clear()
            self._speaking.set()
            chunk_size = max(1, int(sample_rate * self.interrupt_chunk_ms / 1000))
            stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32')
            stream.start()
            completed = True
            try:
                for start in range(0, len(audio_array), chunk_size):
                    if self.interrupt_event.is_set():
                        completed = False
                        break
                    stream.write(audio_array[start:start + chunk_size])
            finally:
                stream.stop()
                stream.close()
                time.sleep(self.post_speech_settle)
                self._speaking.clear()
            return completed

    def play_stream(self, audio_chunks_iter, sample_rate):
        """Returns True if playback completed normally, False if interrupted."""
        with self._lock:
            self.interrupt_event.clear()
            self._speaking.set()
            sub_chunk_size = max(1, int(sample_rate * self.interrupt_chunk_ms / 1000))
            stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32')
            stream.start()
            completed = True
            try:
                for chunk in audio_chunks_iter:
                    if chunk is None:
                        continue
                    for start in range(0, len(chunk), sub_chunk_size):
                        if self.interrupt_event.is_set():
                            completed = False
                            break
                        stream.write(chunk[start:start + sub_chunk_size])
                    if not completed:
                        break
            finally:
                time.sleep(0.3)
                stream.stop()
                stream.close()
                time.sleep(self.post_speech_settle)
                self._speaking.clear()
            return completed

    def wait_until_quiet(self):
        while self.is_speaking:
            time.sleep(0.05)