"""Output guard: model replies are dropped, speech is kept."""

import pytest

from utils import is_model_reply

# Texts Voxtral returned on breath, noise and the bot's echo in agentar-demo, 2026-11-01.
MODEL_REPLIES = [
    "I'm sorry.",
    "Sorry.",
    "I'm sorry?",
    "I'm sorry, but I can't understand that. Could you please\nrepeat or clarify your request?",
    "I'm sorry, I didn't quite catch that. Could you please repeat or clarify your request?",
    "I'm sorry, I didn't understand the question. Could you\nplease repeat it?",
    "I'm sorry, I can't understand that. Could you please repeat or clarify what you need help with?",
    "I'm sorry, but I didn't understand your question. Could you please provide more context or clarify what you need help with?",
    "I'm sorry, could you please repeat or clarify your request?",
    "I'm sorry, I didn't catch that. Can you please repeat your\nquestion?",
    "I'm sorry, but I can't assist with that.",
    "It seems like you've encountered an issue with the audio. Could you please provide more context or clarify your request?",
    "I’m sorry.",
    "The audio is silent.",
    "The speaker says",
    ".",
    "",
    "you",
    "Okay.",
    "End.",
    "Whoa.",
]

SPEECH = [
    "Sorry, a correction. The demo moved to Wednesday at 11.",
    "Sorry, what is that?",
    "I'm sorry for the delay, the build is ready.",
    "Could you please repeat the deadline for the cost report?",
    "Agentar, when do we ship the MVP?",
    "We ship the MVP on November 11.",
    "11.",
    "the",
    "Okay, and we capped the GPU hours at 400 per month.",
    'The transcription of the provided text is: "We decided to ship the MVP on November 10th."',
]


@pytest.mark.parametrize("text", MODEL_REPLIES)
def test_model_reply_is_dropped(text):
    assert is_model_reply(text)


@pytest.mark.parametrize("text", SPEECH)
def test_speech_is_kept(text):
    assert not is_model_reply(text)
