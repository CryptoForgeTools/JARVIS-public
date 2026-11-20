import numpy as np
from faster_qwen3_tts import FasterQwen3TTS
from config.config import (TTS_LANGUAGE, TTS_MODEL, TTS_REFERENCE_AUDIO,
                           TTS_REFERENCE_TEXT, TTS_SPEAKER)

class Speaker:
    def __init__(self, audio_manager, model_name=TTS_MODEL, speaker=TTS_SPEAKER,
                 language=TTS_LANGUAGE, ref_audio=TTS_REFERENCE_AUDIO,
                 ref_text=TTS_REFERENCE_TEXT):
        self.audio = audio_manager
        self.language = language
        self.speaker = speaker
        self.ref_audio = ref_audio if ref_audio and ref_audio.is_file() else None
        self.ref_text = ref_text
        self.tts = FasterQwen3TTS.from_pretrained(model_name)
        if self.ref_audio:
            self.tts.generate_voice_clone(
                text="Warming up.", language=self.language,
                ref_audio=str(self.ref_audio), ref_text=self.ref_text,
            )
        else:
            self.tts.generate_custom_voice(
                text="Warming up.", language=self.language, speaker=self.speaker,
                max_new_tokens=32,
            )

    def generate_audio(self, text):
        if not text:
            return None, None
        try:
            if self.ref_audio:
                audio, sample_rate = self.tts.generate_voice_clone(
                    text=text, language=self.language,
                    ref_audio=str(self.ref_audio), ref_text=self.ref_text,
                )
            else:
                audio, sample_rate = self.tts.generate_custom_voice(
                    text=text, language=self.language, speaker=self.speaker,
                )
            audio_array = np.array(audio[0], dtype=np.float32)
            pad = np.zeros(int(0.15 * sample_rate), dtype=np.float32)
            return np.concatenate([pad, audio_array, pad]), sample_rate
        except Exception as e:
            print(f"TTS generation failed: {e}")
            return None, None

    def speak(self, text, mood="normal"):
        audio_array, sample_rate = self.generate_audio(text)
        self.audio.play(audio_array, sample_rate)
