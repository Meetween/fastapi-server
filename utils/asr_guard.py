"""
Output guard for ASR text: tell a model reply from a transcript.

On near silent or garbled audio, Voxtral answers like a chatbot instead of transcribing: it
apologises, asks to repeat, describes the audio, or emits a lone filler. Such a text is never
shown or stored. A text is a model reply when, once the known phrases and fillers below are taken
out, nothing is left. "Sorry, a correction: the demo moved to Wednesday" keeps its content and stays.
"""

import re

# Phrases the model emits instead of a transcript, on normalised text: lower case, straight
# apostrophes, no punctuation. Measured on the agentar-demo recordings and the eval set.
MODEL_REPLY_PHRASES = (
    # Apologies.
    r"(?:i'm|i am|im) (?:so |very |really )?sorry",
    r"sorry",
    # Cannot follow.
    r"i (?:can't|cannot|can not|couldn't|could not|didn't|did not|don't|do not|wasn't able to|was not able to)"
    r"(?: quite)? (?:understand|catch|hear|get|make out|follow)"
    r"(?: (?:that|this|it|you|what you said|what you need help with|what you mean"
    r"|the question|your question|the audio|your request))?",
    r"i (?:can't|cannot|can not) (?:assist|help) (?:you )?with (?:that|this)",
    # Requests to repeat, rephrase or clarify.
    r"(?:could|can|would) you (?:please )?(?:repeat|rephrase|clarify|say)"
    r"(?: (?:that|it|this|your question|the question|your request|what you said))?(?: again)?",
    r"(?:could|can|would) you (?:please )?provide (?:more |some )?(?:context|details|information)",
    r"(?:please )?(?:repeat|rephrase|clarify)"
    r"(?: (?:that|it|your question|the question|your request|what you need help with|what you mean))?",
    # Descriptions of the audio instead of its words.
    r"the transcription(?: of the(?: provided)? (?:text|audio))? is",
    r"the speaker says",
    r"it seems (?:like|that) you've encountered an issue with the audio",
    r"the audio(?: (?:is|seems|appears)(?: to be)? (?:silent|empty|unclear|inaudible|not clear))?",
    r"there is no (?:speech|audio|sound)(?: in the audio)?",
)

# Words left over from a reply, and the lone fillers the model emits on noise.
FILLERS = frozenset({"but", "and", "or", "so", "please", "you", "okay", "end", "whoa", "um", "uh"})

_REPLY = re.compile(r"\b(?:" + "|".join(MODEL_REPLY_PHRASES) + r")\b")
_APOSTROPHES = re.compile(r"[‘’ʼ`]")
_PUNCTUATION = re.compile(r"[^\w\s']+")


def normalise(text: str) -> str:
    text = _APOSTROPHES.sub("'", text.lower())
    return " ".join(_PUNCTUATION.sub(" ", text).split())


def is_model_reply(text: str) -> bool:
    """True when the text holds only model reply phrases and fillers, or nothing at all."""
    rest = _REPLY.sub(" ", normalise(text))
    return all(word in FILLERS for word in rest.split())
