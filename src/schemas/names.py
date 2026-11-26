"""Name rules that keep the "(AI)" disclosure honest.

A person name or an agent name flows into the Jitsi display name and the spoken greeting. It must
not be able to fake the disclosure (parentheses, the word "agentar") or hide it (control and
format characters). The contract states these as ECMA-262 patterns; Python's `re` has no `\\p{L}`,
so the same rules are checked with `unicodedata` categories.
"""
import unicodedata

_AGENT_NAME_PUNCT = frozenset(" _.'-")
_ASCII_DIGITS = frozenset("0123456789")


def _check_agentar_word(value: str) -> None:
    if "agentar" in value.casefold():
        raise ValueError('must not contain the word "agentar"')


def person_name(value: str) -> str:
    """first_name, last_name: no parentheses, no control or format characters, not "agentar"."""
    for ch in value:
        if ch in "()" or unicodedata.category(ch) in {"Cc", "Cf"}:
            raise ValueError("must not contain parentheses or control characters")
    _check_agentar_word(value)
    return value


def agent_name(value: str) -> str:
    """Unicode letters and marks, ASCII digits, space and `_ . ' -`; the first character is a
    letter, mark or digit; 1 to 32 characters; not "agentar"."""
    if not 1 <= len(value) <= 32:
        raise ValueError("must be 1 to 32 characters")

    def word_char(ch: str) -> bool:
        return unicodedata.category(ch)[0] in {"L", "M"} or ch in _ASCII_DIGITS

    if not word_char(value[0]):
        raise ValueError("must start with a letter or digit")
    if not all(word_char(ch) or ch in _AGENT_NAME_PUNCT for ch in value[1:]):
        raise ValueError("may contain letters, digits, space and _ . ' - only")
    _check_agentar_word(value)
    return value


def bot_display_name(value: str) -> str:
    """Display name sent to the worker, "<Name> Agentar": Unicode letters and marks, ASCII digits,
    space and `_ . ' -`, no parentheses, 1 to 48 characters. The word "agentar" is part of it, so
    unlike `agent_name` it is not refused."""
    if not 1 <= len(value) <= 48:
        raise ValueError("must be 1 to 48 characters")
    first, rest = value[0], value[1:]
    if not (unicodedata.category(first)[0] in {"L", "M"} or first in _ASCII_DIGITS):
        raise ValueError("must start with a letter or digit")
    for ch in rest:
        if not (unicodedata.category(ch)[0] in {"L", "M"} or ch in _ASCII_DIGITS or ch in _AGENT_NAME_PUNCT):
            raise ValueError("may contain letters, digits, space and _ . ' - only")
    return value
