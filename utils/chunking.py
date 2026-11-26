"""Chunks for BM25 retrieval: about 150 tokens of text that never crosses a page or a section.

Input is the `Block`s of `utils.text_extract`. Rules:

- A chunk holds blocks of one page and one section only. A new page or a new heading starts a new chunk.
- A section's heading is the first line of every chunk under it, so a chunk that holds only
  `Trento | 60 | 1 December 2026` is still found by a question about "Atlas pilot sites".
- Blocks are packed greedily up to the budget. The budget is `target_tokens` tokens at four
  characters per token, heading included. There is no tokenizer in the API image, and BM25 does not
  need an exact count.
- A block larger than the room left is split at line ends, then at sentence ends, then at words.
- An atomic block (a docx table row) is never split, so the cells of a row stay together. Only a
  row longer than four times the budget is treated as ordinary text.
- No overlap between chunks: the facts of the demo files are one line each.

Chunk numbers `n` start at 0 and follow the document order. `page` is 1 based, absent for txt, md and docx.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from src.database.models.agentar_file import FileChunk
from utils.text_extract import Block

CHARS_PER_TOKEN = 4
MAX_HEADING_CHARS = 100
ATOMIC_OVERSIZE_FACTOR = 4

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


def _clip_heading(section: Optional[str]) -> Optional[str]:
    if not section:
        return None
    return section if len(section) <= MAX_HEADING_CHARS else section[: MAX_HEADING_CHARS - 1].rstrip() + "…"


def _hard_split(text: str, room: int) -> list[str]:
    return [text[start : start + room] for start in range(0, len(text), room)]


def _split_words(text: str, room: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for word in text.split():
        if len(word) > room:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(word, room))
            continue
        candidate = f"{current} {word}" if current else word
        if len(candidate) > room:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _split_sentences(text: str, room: int) -> list[str]:
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        pieces.extend([sentence] if len(sentence) <= room else _split_words(sentence, room))
    return pieces


def _units(block: Block, room: int) -> list[str]:
    """Pieces of one block, each at most `room` characters, except an atomic block that is kept whole."""
    text = block.text
    if len(text) <= room or (block.atomic and len(text) <= room * ATOMIC_OVERSIZE_FACTOR):
        return [text]
    pieces: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        pieces.extend([line] if len(line) <= room else _split_sentences(line, room))
    return pieces


def chunk_blocks(blocks: Sequence[Block], *, target_tokens: int = 150) -> list[FileChunk]:
    budget = max(target_tokens, 1) * CHARS_PER_TOKEN
    chunks: list[FileChunk] = []
    group: Optional[tuple[Optional[int], Optional[str]]] = None
    heading: Optional[str] = None
    current: list[str] = []
    length = 0
    room = budget

    def flush() -> None:
        nonlocal current, length
        if not current:
            return
        body = "\n".join(current)
        text = f"{heading}\n{body}" if heading else body
        chunks.append(FileChunk(n=len(chunks), text=text, page=group[0] if group else None))
        current = []
        length = 0

    for block in blocks:
        key = (block.page, block.section)
        if key != group:
            flush()
            group = key
            heading = _clip_heading(block.section)
            room = max(budget - (len(heading) + 1 if heading else 0), budget // 4)
        for unit in _units(block, room):
            added = len(unit) + (1 if current else 0)
            if current and length + added > room:
                flush()
                added = len(unit)
            current.append(unit)
            length += added
    flush()
    return chunks
