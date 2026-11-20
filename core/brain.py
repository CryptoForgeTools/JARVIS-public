import ollama
from config.config import OLLAMA_MODEL, USER_DETAILS, USER_NAME

MODEL = OLLAMA_MODEL
SYSTEM_PROMPT = f"""You are JARVIS, a personal AI assistant for {USER_NAME}. {USER_DETAILS}

CRITICAL RULES:
- You have NO tools, NO smart home access, NO weather API, NO calendar access, NO ability to control anything physical, and NO internet access right now. You are a text/voice conversation assistant ONLY at this stage.
- NEVER claim to have checked, activated, adjusted, or accessed anything. If asked about weather, smart home devices, calendars, or anything requiring real-world data or action you don't have, say plainly that you don't have that capability yet.
- Do NOT roleplay as the fictional Iron Man/Marvel JARVIS character. Do not call the user "sir," "Master Stark," or reference a "penthouse," "Stark Industries," or any Iron Man lore. You are a real, separate assistant with your own identity, not a character performance.
- Be helpful, concise, confident, and occasionally witty — but always honest about what you can and cannot currently do.
- If you don't know something or can't do something, say so directly instead of inventing a plausible-sounding answer.
"""

ADDRESSEE_CHECK_PROMPT = """You are a classifier for a voice assistant named JARVIS. JARVIS is actively listening during a session, and just captured a transcribed utterance. Decide whether it is a message DIRECTED AT JARVIS (a command, question, or request for the assistant) or NOT directed at JARVIS (e.g. part of a conversation with another person in the room, addressing someone else by name, or side talk unrelated to the assistant).

Respond with EXACTLY one word: YES if it is directed at JARVIS, or NO if it is not. No other text, no punctuation, no explanation.

Utterance: "{utterance}"
"""


class Brain:
    def __init__(self):
        self.base_system_prompt = SYSTEM_PROMPT
        self.history = [{"role": "system", "content": self.base_system_prompt}]

    def reset_history(self, memory_context: str = ""):
        """Call at the start of each new session. Wipes conversation
        history so it doesn't accumulate unbounded across every session
        for the process's entire lifetime (which would eventually blow
        past num_ctx regardless of anything else), and re-seeds the
        system prompt with current known facts about the user -- so
        cross-session continuity comes from the memory file, not from
        an ever-growing raw history."""
        system_content = self.base_system_prompt
        if memory_context:
            system_content += f"\n\n{memory_context}"
        self.history = [{"role": "system", "content": system_content}]

    def ask(self, prompt):
        self.history.append({"role": "user", "content": prompt})

        try:
            response = ollama.chat(
                model=MODEL,
                messages=self.history,
                options={"temperature": 0.4, "num_ctx": 2048, "num_gpu": 999},  # <-- added here
            )
        except ollama.ResponseError as e:
            self.history.pop()
            if "not found" in str(e).lower():
                return f"I can't reach my brain — the model '{MODEL}' isn't pulled. Try: ollama pull {MODEL}"
            return f"My brain hit an error: {e}"
        except ConnectionError:
            self.history.pop()
            return "I can't reach Ollama at all — is the Ollama server running?"
        except Exception as e:
            self.history.pop()
            return f"Something went wrong talking to my brain: {e}"

        reply = response["message"]["content"]
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def ask_stream(self, prompt):
        """Yields complete sentences as they become available, instead of the whole reply at once."""
        self.history.append({"role": "user", "content": prompt})
        full_reply = ""
        buffer = ""

        try:
            stream = ollama.chat(
                model=MODEL,
                messages=self.history,
                stream=True,
                options={"temperature": 0.4, "num_ctx": 2048, "num_gpu": 999},  # <-- added here too
            )
        except Exception as e:
            self.history.pop()
            yield f"My brain hit an error: {e}"
            return

        try:
            for chunk in stream:
                token = chunk["message"]["content"]
                buffer += token
                full_reply += token

                for end_char in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                    if end_char in buffer:
                        idx = buffer.index(end_char) + 1
                        sentence = buffer[:idx].strip()
                        buffer = buffer[idx:]
                        if sentence:
                            yield sentence
                        break
        except Exception as e:
            yield f"My brain hit an error mid-response: {e}"

        if buffer.strip():
            yield buffer.strip()

        self.history.append({"role": "assistant", "content": full_reply})

    def is_addressed_to_me(self, utterance: str) -> bool:
        """Fast, stateless classification -- does NOT touch self.history,
        so a wrong call can't pollute the real conversation context.
        Fails OPEN (returns True) on any error: if the classifier itself
        breaks, it's better to occasionally respond to something not
        meant for it than to go completely deaf to real commands."""
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "user", "content": ADDRESSEE_CHECK_PROMPT.format(utterance=utterance)}
                ],
                options={"temperature": 0.0, "num_ctx": 512, "num_gpu": 999},
            )
            answer = response["message"]["content"].strip().upper()
            return answer.startswith("Y")
        except Exception as e:
            print(f"[brain] addressee check failed, defaulting to YES: {e}")
            return True