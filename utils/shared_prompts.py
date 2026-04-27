from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptBundle:
    """
    Centralized, route-aware prompts.

    Keep these prompts model-agnostic; Triton backends decide how to apply them
    (plain prompt vs JSON envelope, system prompt selection, etc.).
    """

    language: str = "en-US"
    caption_chars: int = 60

    def _lang_hint(self) -> str:
        lang = (self.language or "").strip()
        return f"Language: {lang}" if lang else "Language: en-US"

    def transcribe(self, *, verbatim: bool = True, extra: str | None = None) -> str:
        style = (
            "Verbatim: keep fillers (um/uh), repetitions, and false starts."
            if verbatim
            else "Clean: remove fillers and obvious stutters; keep meaning unchanged."
        )
        base = f"""Transcribe the audio.
- Output ONLY the transcript text.
- Do NOT summarize.
- Do NOT answer questions from speakers.
- Do NOT add explanations, headings, timestamps, or speaker labels unless explicitly asked.
- {style}
- {self._lang_hint()}
"""
        if extra and extra.strip():
            base += f"\nAdditional instruction:\n{extra.strip()}\n"
        return base.strip()

    def caption(self, *, extra: str | None = None) -> str:
        target = max(32, min(120, int(self.caption_chars)))
        base = f"""Generate live captions for the audio.
- Output ONLY caption text (no summary, no commentary).
- Split into short lines suitable for subtitles.
- Prefer ~{target} characters per line.
- Do NOT include timestamps of any kind (examples to avoid: "00:01", "[00:01]", "(00:01)", "00:01.230", "00:01 --> 00:02").
- Do NOT answer questions from speakers; only transcribe what is said.
- Do NOT add speaker labels unless explicitly asked.
- {self._lang_hint()}
"""
        if extra and extra.strip():
            base += f"\nAdditional instruction:\n{extra.strip()}\n"
        return base.strip()

    def summarize(self, *, extra: str | None = None) -> str:
        base = f"""Summarize the audio.
- Write 6–12 bullet points.
- Include: key decisions, action items (with owners if mentioned), dates/numbers, and risks.
- Be faithful to what was actually said; do NOT invent details.
- If something is unclear, say \"unclear\".
- {self._lang_hint()}
"""
        if extra and extra.strip():
            base += f"\nAdditional instruction:\n{extra.strip()}\n"
        return base.strip()

    def chat(
        self,
        *,
        user_question: str,
        meeting_transcript: str | None = None,
        chat_history: str | None = None,
        extra: str | None = None,
    ) -> str:
        """
        Chat/Q&A prompt that is safe for meeting assistants.

        - meeting_transcript: meeting context (may be empty)
        - chat_history: optional formatted history text
        - extra: any additional instruction (rare; keep short)
        """
        base = f"""You are a meeting assistant.
- Answer the user's question using the meeting transcript when available.
- If the transcript does not contain the answer, say you don't know based on the transcript.
- Be concise, specific, and helpful.
- Do NOT invent names, decisions, numbers, or quotes.
- If asked to summarize, prefer bullet points.
- {self._lang_hint()}

MEETING TRANSCRIPT:
{(meeting_transcript or '').strip()}

CHAT HISTORY:
{(chat_history or '').strip()}

USER QUESTION:
{(user_question or '').strip()}
"""
        if extra and extra.strip():
            base += f"\nAdditional instruction:\n{extra.strip()}\n"
        return base.strip()

    def agenda_tracking(
        self,
        *,
        meeting_transcript: str,
        agenda_items: str,
        allowed_statuses: str,
        json_schema: str,
    ) -> str:
        """
        Prompt for agenda status tracking that requests STRICT JSON output.
        """
        return f"""You are an assistant that updates agenda item statuses based on the meeting transcript.

Rules:
- You MUST output ONLY valid JSON matching the provided schema. No markdown, no commentary.
- You MUST keep the same items and indices; only update the \"status\" field.
- Allowed statuses: {allowed_statuses}
- If an item was not discussed, keep it as \"pending\".
- Be conservative: only mark \"completed\" when clearly completed; mark \"skipped\" only when explicitly skipped.

MEETING TRANSCRIPT:
{(meeting_transcript or '').strip()}

AGENDA ITEMS:
{(agenda_items or '').strip()}

OUTPUT JSON SCHEMA (must follow exactly):
{(json_schema or '').strip()}
""".strip()

