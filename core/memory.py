"""
Tier 1 persistent memory for JARVIS: a flat JSON file of durable facts,
no embeddings, no vector search. This is deliberately the simplest
thing that could work, matching the same "start basic" call already
made for the STT/TTS stack — it gets replaced with smarter retrieval
only once this file actually outgrows the 2048-token context budget in
practice, not preemptively.

Two operations:
  - get_context_string(): loads ALL facts and formats them for
    injection into a turn. Fine while the file stays small.
  - distill_session(transcript): at the end of a session, calls Ollama
    DIRECTLY (bypassing Brain's conversational personality wrapper) to
    extract clean, bullet-point facts from what was just discussed,
    then appends the new ones.

File lives at memory/memory.json, next to main.py.
"""

import os
import json
import requests
from config.config import MEMORY_PATH, OLLAMA_MODEL, OLLAMA_URL

_MEMORY_PATH = str(MEMORY_PATH)
_OLLAMA_URL = OLLAMA_URL.rstrip("/") + "/api/generate"
_MAX_FACTS = 200  # simple cap so the file can't grow unbounded before Tier 2 retrieval exists

_DISTILL_PROMPT_TEMPLATE = """You are extracting durable facts to remember about a user, from a voice assistant session transcript below.

Rules:
- Only extract facts worth remembering across FUTURE sessions: preferences, ongoing projects, names, decisions, recurring context.
- Do NOT extract small talk, one-off requests, or anything generic/obvious.
- Output ONLY a plain list, one fact per line, no numbering, no extra commentary.
- If nothing is worth remembering, output nothing at all.

Transcript:
{transcript}

Facts worth remembering:"""


def _ensure_dir():
    os.makedirs(os.path.dirname(_MEMORY_PATH), exist_ok=True)
    if not os.path.exists(_MEMORY_PATH):
        with open(_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)


def load_memory() -> list:
    _ensure_dir()
    if not os.path.exists(_MEMORY_PATH):
        return []
    try:
        with open(_MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_memory(facts: list):
    _ensure_dir()
    with open(_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)


def append_facts(new_facts: list):
    if not new_facts:
        return
    facts = load_memory()
    for fact in new_facts:
        fact = fact.strip("-• \t")
        if fact and fact not in facts:
            facts.append(fact)
    # simple recency cap -- no relevance ranking yet, that's Tier 2's job
    # once this actually becomes a real problem
    facts = facts[-_MAX_FACTS:]
    save_memory(facts)


def get_context_string() -> str:
    facts = load_memory()
    if not facts:
        return ""
    bullet_list = "\n".join(f"- {fact}" for fact in facts)
    return f"Known context about the user:\n{bullet_list}"


def distill_session(transcript: str):
    """Call this once, at the END of a session, with the full session
    transcript. Best run in a background thread -- it's a blocking
    network call to Ollama and shouldn't delay returning to wake-word
    listening."""
    if not transcript.strip():
        return

    prompt = _DISTILL_PROMPT_TEMPLATE.format(transcript=transcript)
    try:
        response = requests.post(
            _OLLAMA_URL,
            json={"model": _OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
    except requests.RequestException as e:
        print(f"[memory] distillation failed (Ollama unreachable?): {e}")
        return

    new_facts = [line.strip() for line in raw.splitlines() if line.strip()]
    if new_facts:
        print(f"[memory] learned {len(new_facts)} new fact(s) this session")
        append_facts(new_facts)