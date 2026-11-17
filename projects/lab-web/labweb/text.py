"""Tokenisation shared with the training code (lower-case words and ! ?)."""

import re

_TOKEN = re.compile(r"[a-z0-9']+|[!?]")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("<br />", " "))


def encode(tokens: list[str], stoi: dict[str, int], max_len: int = 400) -> list[int]:
    """Words -> ids (unknown = 1), truncated to max_len, never empty."""
    return [stoi.get(w, 1) for w in tokens[:max_len]] or [1]
