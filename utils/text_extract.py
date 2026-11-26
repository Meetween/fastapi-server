"""Text extraction for knowledge base files: txt, md, pdf and docx.

Output is a list of `Block`s (a paragraph, a table row, a page paragraph) that
`utils.chunking.chunk_blocks` packs into chunks. Nothing here reads the network or a database.

pdf and docx are parsed by code that has had parser bombs and denial of service advisories
(A3 H3), so they never run inside the API process. `run_isolated` starts this very file as a
script in a fresh interpreter (`python -I`, empty environment, no package imports) with:

- `RLIMIT_AS` (memory), `RLIMIT_CPU`, `RLIMIT_FSIZE=0` (no file writes), `RLIMIT_CORE=0`,
  set by the child before it opens the input;
- a wall clock timeout enforced by the parent, which kills the child;
- the input on stdin and a JSON result on stdout, so the parent never unpickles anything
  that the parser touched.

One process per file, not a pool: `RLIMIT_CPU` counts the whole life of a process, so a
long lived worker would run out of CPU budget after a few files and a hung worker could not be
killed without reaching into executor internals. Start up costs about 0.3 s, paid by an upload
only. At most one extraction runs at a time (`_SLOT`).

On Linux a limit that cannot be applied fails the extraction. On other systems (a developer
laptop: macOS refuses `RLIMIT_AS`) the file is still parsed with the CPU and wall limits and the
parent logs a warning.

The top of this module must stay standard library only: it is executed as a script and must not
import the `utils` package, which loads audio and model libraries.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT = "text/plain"
MARKDOWN = "text/markdown"
SUPPORTED_TYPES = (TEXT, MARKDOWN, PDF, DOCX)

# docx archive pre check (A3 H3), applied before python-docx opens the file.
MAX_ZIP_ENTRIES = 500
MAX_ZIP_TOTAL_BYTES = 50 * 1024 * 1024
MAX_ZIP_RATIO = 100
# Entries below this size are not ratio checked: a small part that compresses 150 to 1 is
# harmless, and the entry count and total caps already bound them.
RATIO_CHECK_FROM_BYTES = 64 * 1024
REQUIRED_DOCX_PARTS = ("[Content_Types].xml", "word/document.xml")

MAX_PDF_PAGES = 200
MAX_TABLE_DEPTH = 3

LIMIT_AS_BYTES = 1024 * 1024 * 1024
LIMIT_CPU_SECONDS = 30
WALL_SECONDS = 30
# Result of one extraction can hold 200k characters, a few hundred KB of JSON. Anything over
# this is not our child talking.
MAX_RESULT_BYTES = 8 * 1024 * 1024

_SCRIPT = Path(__file__).resolve()
_SLOT = threading.Semaphore(1)
_CHILD_ENV = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "PYTHONUTF8": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


@dataclass(frozen=True)
class Block:
    """One unit of extracted text. `atomic` blocks (table rows) are never split by the chunker."""

    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    atomic: bool = False


@dataclass(frozen=True)
class Extraction:
    status: str  # ok, empty, truncated, failed
    blocks: list[Block] = field(default_factory=list)
    chars: int = 0
    error: Optional[str] = None


def _failed(message: str) -> Extraction:
    return Extraction(status="failed", error=message)


# docx archive pre check ------------------------------------------------------------------------


class DocxArchiveError(ValueError):
    """The file is not a docx we are willing to open. The message is for logs, not for users."""


def check_docx_archive(data: bytes) -> None:
    """Refuse zip bombs and non docx archives before any parser sees the bytes.

    Entry count from the end of central directory record first, so a hostile directory is never
    materialised; then entry count, summed uncompressed size and per entry ratio from the
    directory; then the two parts every docx has. Declared sizes are enforced by `zipfile` when
    an entry is read, so a lying directory fails the read instead of expanding.
    """
    if not data.startswith(b"PK\x03\x04"):
        raise DocxArchiveError("not a zip archive")
    tail = data[-(65535 + 22) :]
    at = tail.rfind(b"PK\x05\x06")
    if at < 0 or len(tail) - at < 22:
        raise DocxArchiveError("no end of central directory")
    total_entries = struct.unpack_from("<H", tail, at + 10)[0]
    if total_entries > MAX_ZIP_ENTRIES:
        raise DocxArchiveError(f"{total_entries} entries")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError, ValueError, NotImplementedError) as exc:
        raise DocxArchiveError("unreadable archive") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise DocxArchiveError(f"{len(infos)} entries")
        total = 0
        names = set()
        for info in infos:
            total += info.file_size
            if total > MAX_ZIP_TOTAL_BYTES:
                raise DocxArchiveError("uncompressed size over the cap")
            if info.file_size >= RATIO_CHECK_FROM_BYTES and info.file_size > MAX_ZIP_RATIO * max(info.compress_size, 1):
                raise DocxArchiveError("compression ratio over the cap")
            names.add(info.filename)
        for required in REQUIRED_DOCX_PARTS:
            if required not in names:
                raise DocxArchiveError(f"missing {required}")


# Collecting blocks ----------------------------------------------------------------------------

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]")
_HSPACE = re.compile(r"[ \t   ]+")
_BLANKS = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Newlines normalised, control characters and lone surrogates dropped (BSON refuses them),
    runs of blanks collapsed, lines stripped."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", "\n").replace(" ", "\n\n")
    text = _CONTROL.sub("", text)
    text = _HSPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


class _Collector:
    """Accumulates blocks up to `max_chars` of text and remembers whether it cut anything."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        self.chars = 0
        self.blocks: list[Block] = []
        self.truncated = False

    def add(self, text: str, *, page: Optional[int] = None, section: Optional[str] = None, atomic: bool = False) -> bool:
        """Add a block. False once text had to be cut: the caller stops reading."""
        text = text.strip()
        if not text:
            return True
        room = self.max_chars - self.chars
        if len(text) > room:
            text = text[:room].rstrip()
            self.truncated = True
        if text:
            self.blocks.append(Block(text=text, page=page, section=section, atomic=atomic))
            self.chars += len(text)
        return not self.truncated

    def result(self, *, cut: bool = False) -> Extraction:
        truncated = self.truncated or cut
        if not self.blocks:
            return Extraction(status="empty")
        return Extraction(status="truncated" if truncated else "ok", blocks=self.blocks, chars=self.chars)


# txt and md ----------------------------------------------------------------------------------

_HEADING = re.compile(r"^\s{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^\s{0,3}(```|~~~)")


def extract_plain(data: bytes, *, markdown: bool, max_chars: int) -> Extraction:
    """Paragraphs split on blank lines. In markdown the nearest heading becomes the section of the
    blocks under it; headings inside code fences are text."""
    collector = _Collector(max_chars)
    text = clean_text(data.decode("utf-8", errors="replace").lstrip("﻿"))
    section: Optional[str] = None
    paragraph: list[str] = []
    in_fence = False

    def flush() -> bool:
        if not paragraph:
            return True
        joined = "\n".join(paragraph)
        paragraph.clear()
        return collector.add(joined, section=section)

    for line in text.split("\n"):
        if markdown and _FENCE.match(line):
            in_fence = not in_fence
        heading = _HEADING.match(line) if markdown and not in_fence else None
        if heading:
            if not flush():
                break
            section = heading.group(1).strip() or None
        elif not line.strip():
            if not flush():
                break
        else:
            paragraph.append(line)
    else:
        flush()
    return collector.result()


# pdf -----------------------------------------------------------------------------------------


def extract_pdf(data: bytes, *, max_chars: int) -> Extraction:
    """Text layer of the first `MAX_PDF_PAGES` pages, one block group per page. No OCR: a scanned
    pdf has no text layer and ends as `empty`."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data), strict=False)
    if reader.is_encrypted and not reader.decrypt(""):
        return _failed("The pdf is password protected")
    collector = _Collector(max_chars)
    total_pages = len(reader.pages)
    for index in range(min(total_pages, MAX_PDF_PAGES)):
        page_text = clean_text(reader.pages[index].extract_text() or "")
        for paragraph in re.split(r"\n\s*\n", page_text):
            if not collector.add(paragraph, page=index + 1):
                return collector.result()
    return collector.result(cut=total_pages > MAX_PDF_PAGES)


# docx ----------------------------------------------------------------------------------------


def _cell_text(cell: Any, depth: int) -> str:
    """Paragraphs of a cell joined by a space; nested tables as rows joined by ` ; `."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parts: list[str] = []
    for item in cell.iter_inner_content():
        if isinstance(item, Paragraph):
            parts.append(item.text)
        elif isinstance(item, Table) and depth < MAX_TABLE_DEPTH:
            parts.extend(row for row in _table_rows(item, depth + 1))
    return " ".join(" ".join(part.split()) for part in parts if part.strip())


def _table_rows(table: Any, depth: int) -> Iterable[str]:
    """Each row as `a | b | c`. A merged cell appears once, empty cells are dropped."""
    for row in table.rows:
        seen: set[int] = set()
        cells: list[str] = []
        for cell in row.cells:
            marker = id(cell._tc)
            if marker in seen:
                continue
            seen.add(marker)
            text = _cell_text(cell, depth)
            if text:
                cells.append(text)
        if cells:
            yield " | ".join(cells)


def _is_heading(paragraph: Any) -> bool:
    try:
        name = paragraph.style.name or ""
    except Exception:  # a dangling style reference is a normal paragraph
        return False
    return name == "Title" or name.startswith("Heading")


def extract_docx(data: bytes, *, max_chars: int) -> Extraction:
    """Body paragraphs and tables in document order. A heading starts a section. Each table row is
    one atomic block, so the cells of a row stay together: `Trento | 60 | 1 December 2026`.
    Headers, footers, footnotes and text boxes are not read."""
    check_docx_archive(data)
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(io.BytesIO(data))
    collector = _Collector(max_chars)
    section: Optional[str] = None
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = clean_text(item.text)
            if not text:
                continue
            if _is_heading(item):
                section = " ".join(text.split())
                continue
            if not collector.add(text, section=section):
                break
        elif isinstance(item, Table):
            stop = False
            for row in _table_rows(item, 1):
                if not collector.add(clean_text(row), section=section, atomic=True):
                    stop = True
                    break
            if stop:
                break
    return collector.result()


# Child process -------------------------------------------------------------------------------


def _apply_limits(as_bytes: int, cpu_seconds: int) -> bool:
    """Called first in the child. Returns whether the memory limit is in force."""
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
        return True
    except (ValueError, OSError):
        if sys.platform.startswith("linux"):
            raise
        return False


def _child_main(argv: list[str]) -> int:
    kind, max_chars, as_bytes, cpu_seconds = argv[1], int(argv[2]), int(argv[3]), int(argv[4])
    memory_limited = _apply_limits(as_bytes, cpu_seconds)
    data = sys.stdin.buffer.read()
    exception_name = ""
    try:
        if kind == "pdf":
            result = extract_pdf(data, max_chars=max_chars)
        elif kind == "docx":
            result = extract_docx(data, max_chars=max_chars)
        else:
            result = _failed("Unsupported kind")
    except DocxArchiveError:
        result = _failed("The docx archive was refused")
        exception_name = "DocxArchiveError"
    except MemoryError:
        result = _failed("The file needs too much memory to read")
        exception_name = "MemoryError"
    except Exception as exc:
        result = _failed(f"The {kind} could not be read")
        exception_name = type(exc).__name__
    payload = {"result": asdict(result), "memory_limited": memory_limited, "exception": exception_name}
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()
    return 0


# Parent side ---------------------------------------------------------------------------------


_warned_no_memory_limit = False


def _warn_no_memory_limit() -> None:
    global _warned_no_memory_limit
    if not _warned_no_memory_limit:
        _warned_no_memory_limit = True
        logger.warning("RLIMIT_AS is not available on this system, extraction runs without a memory limit")


def _decode_result(raw: bytes, max_chars: int) -> Extraction:
    payload = json.loads(raw)
    data = payload["result"]
    status = data["status"]
    if status not in {"ok", "empty", "truncated", "failed"}:
        raise ValueError("bad status")
    blocks = []
    chars = 0
    for item in data["blocks"]:
        block = Block(
            text=str(item["text"]),
            page=int(item["page"]) if item["page"] is not None else None,
            section=str(item["section"]) if item["section"] is not None else None,
            atomic=bool(item["atomic"]),
        )
        chars += len(block.text)
        blocks.append(block)
    if chars > max_chars:
        raise ValueError("result over the cap")
    error = data["error"] if status == "failed" else None
    if payload.get("exception"):
        logger.info("Extraction child ended with %s", payload["exception"])
    if payload.get("memory_limited") is False:
        _warn_no_memory_limit()
    return Extraction(status=status, blocks=blocks, chars=chars, error=str(error) if error else None)


def run_isolated(
    kind: str,
    data: bytes,
    *,
    max_chars: int,
    as_bytes: int = LIMIT_AS_BYTES,
    cpu_seconds: int = LIMIT_CPU_SECONDS,
    wall_seconds: float = WALL_SECONDS,
    script: Path = _SCRIPT,
) -> Extraction:
    """Blocking. Run it in a thread. Never raises for a bad file: any failure is `failed`."""
    if not _SLOT.acquire(timeout=wall_seconds * 2):
        return _failed("The extractor is busy")
    try:
        command = [sys.executable, "-I", str(script), kind, str(max_chars), str(as_bytes), str(cpu_seconds)]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_CHILD_ENV,
            cwd=tempfile.gettempdir(),
            close_fds=True,
        )
        try:
            out, _ = process.communicate(data, timeout=wall_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            logger.warning("Extraction of a %s killed after %.0f s", kind, wall_seconds)
            return _failed("Extraction took too long")
        except BaseException:
            process.kill()
            process.wait()
            raise
        if not out or len(out) > MAX_RESULT_BYTES:
            logger.warning("Extraction of a %s ended with code %s and no result", kind, process.returncode)
            return _failed("Extraction exceeded its limits")
        try:
            return _decode_result(out, max_chars)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Extraction of a %s returned an unusable result: %s", kind, type(exc).__name__)
            return _failed("Extraction failed")
    finally:
        _SLOT.release()


_KINDS = {PDF: "pdf", DOCX: "docx"}


async def extract_text(content_type: str, data: bytes, *, max_chars: int) -> Extraction:
    """Text of a stored file. txt and md are read here; pdf and docx in a limited process."""
    if content_type in (TEXT, MARKDOWN):
        return await asyncio.to_thread(
            extract_plain, data, markdown=content_type == MARKDOWN, max_chars=max_chars
        )
    kind = _KINDS.get(content_type)
    if kind is None:
        return _failed("Unsupported type")
    return await asyncio.to_thread(run_isolated, kind, data, max_chars=max_chars)


if __name__ == "__main__":
    sys.exit(_child_main(sys.argv))
