import random

FILLERS = {
    "code": ["Let me look at that code.", "Checking that for you.", "One moment, analyzing."],
    "search": ["Let me check that.", "One second.", "Looking into it."],
    "general": ["Hmm, let me think.", "Give me a second.", "One moment."],
}

def needs_filler(text: str, word_count_threshold: int = 6) -> bool:
    return len(text.split()) > word_count_threshold

def pick_filler(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["code", "debug", "function", "script", "error"]):
        category = "code"
    elif any(w in text_lower for w in ["search", "look up", "find", "check online"]):
        category = "search"
    else:
        category = "general"
    return random.choice(FILLERS[category])