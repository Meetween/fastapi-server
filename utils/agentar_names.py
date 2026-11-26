"""Names the Agentar speaks, and the escalation line built from them.

One source for the dispatch payload (`agentar_worker`) and the chat path (`agentar_prompt`): the
worker compares a chat reply with the `escalation_line` it was given at dispatch, so both sides
must produce the same sentence for the same owner, character for character.
"""
import re
import unicodedata
from typing import Optional

FALLBACK_OWNER = "the meeting owner"
# Owner's full name, no trailing period. The worker matches the reply against this exact text.
ESCALATION_TEMPLATE = "I do not have that, I will check with {owner_full_name} and report back"
# Per name part. A full name is at most two parts and one space.
OWNER_PART_CHARS = 64
OWNER_FULL_CHARS = 2 * OWNER_PART_CHARS + 1

_NAME_PUNCT = " _.'-"
_DIGITS = frozenset("0123456789")


def clean_name(value: Optional[str], limit: int) -> Optional[str]:
    """A name reduced to what a display name allows: letters, marks, digits and ` _ . ' -`.
    Whitespace and control characters become one space, other symbols are dropped, the word
    "agentar" is removed so a name can never fake or double the label."""
    if not value:
        return None
    kept: list[str] = []
    for ch in unicodedata.normalize("NFC", value):
        category = unicodedata.category(ch)
        if category[0] in "LM" or ch in _DIGITS or ch in "_.'-":
            kept.append(ch)
        elif ch == " " or category[0] == "Z" or category == "Cc":
            kept.append(" ")
    text = re.sub(" +", " ", "".join(kept))
    while "agentar" in text.casefold():
        text = re.sub("(?i)agentar", "", text)
    text = re.sub(" +", " ", text).strip(_NAME_PUNCT)
    text = text[:limit].rstrip(_NAME_PUNCT)
    return text or None


def owner_names(first_name: Optional[str], last_name: Optional[str]) -> tuple[str, str]:
    """(full name, first name) as spoken. Without names both are "the meeting owner"."""
    first = clean_name(first_name, OWNER_PART_CHARS)
    last = clean_name(last_name, OWNER_PART_CHARS)
    full = " ".join(part for part in (first, last) if part) or FALLBACK_OWNER
    return full, first or FALLBACK_OWNER


def escalation_line(owner_full_name: str) -> str:
    return ESCALATION_TEMPLATE.format(owner_full_name=owner_full_name)
