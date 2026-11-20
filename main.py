import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import queue
import random
import threading
import time

import cv2

from core.brain import Brain
from core.filler import needs_filler, pick_filler
from core.audio_manager import AudioManager
from voice.wakeword import WakeWordDetector
from voice.listen import Listener
from voice.speak import Speaker
from camera.camera_presence import FaceDetector, PresenceMonitor
from camera.enroll_face import load_recognizer
from listeners.barge_in import BargeInMonitor
from listeners.clap_detector import ClapDetector
from listeners.speaker_id import SpeakerVerifier
from core import memory
from config.config import FACE_CONFIG_PATH, USER_NAME

# Create the ignored local memory file on the first application run.
memory.load_memory()

SESSION_SILENCE_LIMIT = 2
SESSION_LISTEN_TIMEOUT = 8
EXIT_PHRASES = ["exit", "quit", "goodbye", "that's all", "stop listening"]

# Wake-word / clap acknowledgment — picked at random so it's not "Yes?" every single time.
WAKE_RESPONSES = [
    "Yes?",
    "I'm listening.",
    "Go ahead.",
    "What can I do for you?",
    "At your service.",
    f"Yes, {USER_NAME}.",
    "I'm here.",
    "Go on.",
]

# Presence-triggered greetings, split by time of day, also randomized within each bucket.
MORNING_GREETINGS = [
    f"Good morning, {USER_NAME}.",
    "Morning.",
    "Rise and shine.",
]
EVENING_GREETINGS = [
    f"Good evening, {USER_NAME}.",
    "Good evening.",
    "Welcome back.",
]
DAY_GREETINGS = [
    f"Hey, {USER_NAME}.",
    "Welcome back.",
    "Good to see you.",
    "Ah, there you are.",
]

audio_manager = AudioManager(post_speech_settle=0.4)
brain = Brain()
wakeword = WakeWordDetector(threshold=0.25)
listener = Listener()
speaker = Speaker(audio_manager=audio_manager)
try:
    face_detector = FaceDetector()
    face_recognizer = load_recognizer(str(FACE_CONFIG_PATH))
except (FileNotFoundError, cv2.error) as exc:
    print(f"[camera] presence detection disabled: {exc}")
    face_detector = None
    face_recognizer = None
speaker_verifier = SpeakerVerifier()

# Set while waiting for the wake word; cleared during an active session,
# so a presence/clap trigger never talks over an in-progress turn.
is_idle = threading.Event()
is_idle.set()

# Any of wake-word, clap, or camera presence can set this to mean
# "start a session" — the outer loop just waits on it, unaware of which
# source fired. last_wake_source lets the loop pick the right opening
# line (a personalized greeting for presence, plain "Yes?" otherwise).
wake_event = threading.Event()
last_wake_source = ["wakeword"]

# Reset once per session (right after a wake trigger). session_transcript
# collects "User:"/"JARVIS:" lines for end-of-session distillation.
session_transcript = []


def trigger_wake(source):
    if not is_idle.is_set():
        print(f"[wake] {source} trigger ignored — mid-session")
        return  # mid-session — ignore, this trigger just doesn't get a turn
    print(f"[wake] triggered by {source}")
    last_wake_source[0] = source
    wake_event.set()


def on_presence_arrival():
    trigger_wake("presence")


if face_detector is not None and face_recognizer is not None:
    presence_monitor = PresenceMonitor(
        face_detector, face_recognizer, on_arrival=on_presence_arrival,
    )
    threading.Thread(target=presence_monitor.run_forever, daemon=True).start()

clap_detector = ClapDetector(on_wake=lambda: trigger_wake("clap"))
clap_detector.calibrate()
clap_detector.start()

# Set at the start of every handle_turn() call, checked inside its
# generation/TTS loops. A barge-in sets this so the OLD turn's threads
# stop burning GPU/CPU on a response nobody's listening to anymore,
# instead of just muting the speaker while they keep running.
cancel_current_turn = threading.Event()


def on_barge_in():
    # AudioManager.interrupt() has already been called by BargeInMonitor
    # itself by the time this fires — this just stops the upstream
    # generation/TTS threads for the turn that got cut off.
    cancel_current_turn.set()


barge_in_monitor = BargeInMonitor(
    audio_manager,
    on_barge_in=on_barge_in,
)

_calibration_audio, _calibration_sr = speaker.generate_audio(
    "Calibrating microphone, one moment."
)
barge_in_monitor.calibrate_self_bleed(_calibration_audio, _calibration_sr)


def barge_in_watcher():
    """Arms/disarms BargeInMonitor to track AudioManager.is_speaking, so
    every playback path (fillers, 'Yes?', conversational turns) gets
    barge-in coverage without touching each call site individually."""
    was_speaking = False
    while True:
        speaking = audio_manager.is_speaking
        if speaking and not was_speaking:
            barge_in_monitor.arm()
        elif not speaking and was_speaking:
            barge_in_monitor.disarm()
        was_speaking = speaking
        time.sleep(0.05)


threading.Thread(target=barge_in_watcher, daemon=True).start()

print("JARVIS ONLINE")


def handle_turn(user_input):
    """Runs one full brain -> TTS -> playback turn for a single user utterance."""
    print(f"You: {user_input}")

    cancel_current_turn.clear()
    session_transcript.append(f"User: {user_input}")

    if needs_filler(user_input):
        speaker.speak(pick_filler(user_input))  # blocks briefly, fully serialized via audio_manager

    sentence_queue = queue.Queue()
    audio_queue = queue.Queue()
    text_done = threading.Event()
    tts_done = threading.Event()
    reply_sentences = []

    def generate_text():
        for sentence in brain.ask_stream(user_input):
            if cancel_current_turn.is_set():
                break
            sentence_queue.put(sentence)
        text_done.set()

    def generate_tts():
        while not text_done.is_set() or not sentence_queue.empty():
            if cancel_current_turn.is_set():
                break
            try:
                sentence = sentence_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            audio_array, sample_rate = speaker.generate_audio(sentence)
            audio_queue.put((sentence, audio_array, sample_rate))
        tts_done.set()

    threading.Thread(target=generate_text, daemon=True).start()
    threading.Thread(target=generate_tts, daemon=True).start()

    print("JARVIS: ", end="", flush=True)

    # Collect (text-printing side effect + chunk generator) for AudioManager.play_stream
    collected_sample_rate = [None]

    def chunk_generator():
        while not tts_done.is_set() or not audio_queue.empty():
            if cancel_current_turn.is_set():
                return
            try:
                sentence, audio_array, sample_rate = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if audio_array is None:
                continue
            print(sentence, end=" ", flush=True)
            collected_sample_rate[0] = sample_rate
            reply_sentences.append(sentence)
            yield audio_array

    # We need a sample rate before opening the stream, so peek the first chunk
    gen = chunk_generator()
    try:
        first_chunk = next(gen)
    except StopIteration:
        print()
        return

    def full_gen():
        yield first_chunk
        yield from gen

    completed = audio_manager.play_stream(full_gen(), collected_sample_rate[0])
    print()
    if reply_sentences:
        session_transcript.append(f"JARVIS: {' '.join(reply_sentences)}")
    if not completed:
        print("(interrupted by barge-in)")


def wakeword_worker():
    wakeword.listen_for_wakeword()
    trigger_wake("wakeword")


while True:
    is_idle.set()
    audio_manager.wait_until_quiet()
    threading.Thread(target=wakeword_worker, daemon=True).start()
    wake_event.wait()
    wake_event.clear()
    is_idle.clear()

    if last_wake_source[0] == "presence":
        hour = time.localtime().tm_hour
        if hour < 12:
            pool = MORNING_GREETINGS
        elif hour >= 18:
            pool = EVENING_GREETINGS
        else:
            pool = DAY_GREETINGS
        speaker.speak(random.choice(pool))
    else:
        speaker.speak(random.choice(WAKE_RESPONSES))

    session_transcript.clear()
    brain.reset_history(memory.get_context_string())

    empty_count = 0
    session_active = True

    while session_active:
        audio_manager.wait_until_quiet()
        user_input, user_audio = listener.listen_and_transcribe(max_seconds=SESSION_LISTEN_TIMEOUT)

        if not user_input:
            empty_count += 1
            if empty_count >= SESSION_SILENCE_LIMIT:
                print("(session ended — back to wake word)")
                session_active = False
            continue

        if not speaker_verifier.is_me(user_audio):
            print(f"(ignored — not your voice: \"{user_input}\")")
            empty_count += 1
            if empty_count >= SESSION_SILENCE_LIMIT:
                print("(session ended — back to wake word)")
                session_active = False
            continue

        if not brain.is_addressed_to_me(user_input):
            print(f"(ignored — not addressed to JARVIS: \"{user_input}\")")
            empty_count += 1
            if empty_count >= SESSION_SILENCE_LIMIT:
                print("(session ended — back to wake word)")
                session_active = False
            continue

        empty_count = 0

        if user_input.lower().strip(".!? ") in EXIT_PHRASES:
            speaker.speak(f"Good day, {USER_NAME}.")
            session_active = False
            continue

        handle_turn(user_input)

    # Session over -- distill what was discussed into durable facts,
    # in the background so it doesn't delay returning to wake-word mode.
    threading.Thread(
        target=memory.distill_session,
        args=("\n".join(session_transcript),),
        daemon=True,
    ).start()