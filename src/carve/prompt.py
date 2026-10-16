"""Fixed single-reference Ref2VA prompt used by v4 static conditioning."""

import hashlib
from pathlib import Path

STATIC_PROMPT = (
    (Path(__file__).with_name("prompts") / "reference_only_ref2va_v4.txt").read_text().strip()
)


def static_prompt_hash():
    return hashlib.sha256(STATIC_PROMPT.encode()).hexdigest()
