# voice/wakeword.py
import sounddevice as sd
from openwakeword.model import Model
from config.config import WAKEWORD_MODEL

class WakeWordDetector:
    def __init__(self, wakeword_model=WAKEWORD_MODEL, threshold=0.25, sample_rate=16000):
        self.model = Model(wakeword_models=[wakeword_model])
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_size = 1280

    def listen_for_wakeword(self):
        print("Listening for wake word...")
        with sd.InputStream(channels=1, samplerate=self.sample_rate,
                             blocksize=self.chunk_size, dtype='int16') as stream:
            while True:
                audio_chunk, _ = stream.read(self.chunk_size)
                prediction = self.model.predict(audio_chunk.flatten())
                for name, score in prediction.items():
                    if score > self.threshold:
                        print(f"Wake word detected ({name}: {score:.2f})")
                        return