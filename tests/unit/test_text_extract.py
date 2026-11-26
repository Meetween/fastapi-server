"""Text extraction: the E0 knowledge base files, parser bombs, and the limited process.

The limit tests start real child processes. The memory limit needs `RLIMIT_AS`, which macOS
refuses, so that one runs on Linux only (the image and the VM); the CPU, wall clock, write and
environment limits run everywhere.
"""
import io
import struct
import sys
import textwrap
import time
import zipfile
from pathlib import Path

import pytest
from pypdf import PdfWriter

from utils import text_extract
from utils.text_extract import (
    DOCX,
    MARKDOWN,
    MAX_PDF_PAGES,
    MAX_ZIP_ENTRIES,
    PDF,
    TEXT,
    Block,
    DocxArchiveError,
    check_docx_archive,
    clean_text,
    extract_docx,
    extract_pdf,
    extract_plain,
    extract_text,
    run_isolated,
)

KB = Path(__file__).resolve().parents[1] / "data" / "agentar_kb"

DOCX_BODY = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body/></w:document>'
)


def make_zip(entries: dict[str, bytes], *, compress: int = zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compress) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def minimal_docx(extra: dict[str, bytes] | None = None) -> bytes:
    return make_zip({"[Content_Types].xml": b"<Types/>", "word/document.xml": DOCX_BODY, **(extra or {})})


def text_pdf(pages: list[list[str]]) -> bytes:
    """A small valid pdf with one Helvetica line per entry, one list of lines per page."""
    objects: list[bytes] = []
    page_ids = [4 + 2 * n for n in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for number, lines in enumerate(pages):
        content_id = page_ids[number] + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        drawing = "BT /F1 12 Tf 14 TL 50 740 Td " + " ".join(f"({line}) Tj T*" for line in lines) + " ET"
        objects.append(f"<< /Length {len(drawing)} >>\nstream\n{drawing}\nendstream".encode())
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def blank_pdf(pages: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class TestCleanText:
    def test_newlines_blanks_and_controls(self) -> None:
        assert clean_text("a\r\nb\rc\x00d\x07  e\t\tf g") == "a\nb\ncd e f g"

    def test_lone_surrogates_are_dropped_because_bson_refuses_them(self) -> None:
        cleaned = clean_text("ok\ud800 text\udfff")
        assert cleaned == "ok text"
        cleaned.encode("utf-8")

    def test_blank_runs_collapse_to_one_empty_line(self) -> None:
        assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


class TestPlainText:
    def test_paragraphs_split_on_blank_lines(self) -> None:
        result = extract_plain(b"one\ntwo\n\nthree\n\n\n\nfour", markdown=False, max_chars=1000)
        assert [block.text for block in result.blocks] == ["one\ntwo", "three", "four"]
        assert result.status == "ok"

    def test_text_does_not_read_headings(self) -> None:
        result = extract_plain(b"# not a heading here\nbody", markdown=False, max_chars=1000)
        assert result.blocks == [Block(text="# not a heading here\nbody")]

    def test_markdown_sections(self) -> None:
        source = b"intro\n\n# One\ntext a\n\n## Two ##\ntext b\n- item\n\n```\n# not a heading\n```\n"
        result = extract_plain(source, markdown=True, max_chars=1000)
        assert [(block.section, block.text) for block in result.blocks] == [
            (None, "intro"),
            ("One", "text a"),
            ("Two", "text b\n- item"),
            ("Two", "```\n# not a heading\n```"),
        ]

    def test_bom_and_empty(self) -> None:
        assert extract_plain(b"\xef\xbb\xbfhello", markdown=False, max_chars=100).blocks[0].text == "hello"
        empty = extract_plain(b" \n\t\n", markdown=False, max_chars=100)
        assert empty.status == "empty" and empty.blocks == [] and empty.chars == 0

    def test_cap_cuts_and_marks_truncated(self) -> None:
        result = extract_plain(("word " * 100).encode(), markdown=False, max_chars=50)
        assert result.status == "truncated"
        assert result.chars == sum(len(block.text) for block in result.blocks) <= 50

    def test_text_that_ends_exactly_at_the_cap_is_not_truncated(self) -> None:
        result = extract_plain(b"x" * 50, markdown=False, max_chars=50)
        assert (result.status, result.chars) == ("ok", 50)

    def test_cap_across_paragraphs(self) -> None:
        source = b"\n\n".join([b"p" * 30] * 10)
        result = extract_plain(source, markdown=False, max_chars=100)
        assert result.status == "truncated"
        assert result.chars <= 100
        assert len(result.blocks) == 4

    def test_invalid_utf8_is_replaced_not_raised(self) -> None:
        assert extract_plain(b"caf\xe9", markdown=False, max_chars=100).status == "ok"


class TestE0Files:
    def test_docx_keeps_table_rows_together_under_their_heading(self) -> None:
        result = extract_docx((KB / "atlas-sites.docx").read_bytes(), max_chars=200_000)
        assert result.status == "ok"
        rows = [block for block in result.blocks if block.atomic]
        assert [row.text for row in rows] == [
            "Site | Pilot seats | Start",
            "Trento | 60 | 1 December 2026",
            "Bonn | 60 | 1 December 2026",
            "Lyon | 30 | January 2027",
        ]
        assert {row.section for row in rows} == {"Atlas pilot sites"}
        assert {block.section for block in result.blocks if not block.atomic} == {
            "Atlas pilot sites",
            "Contacts and rooms",
        }
        assert all(block.page is None for block in result.blocks)

    def test_pdf_pages_are_kept(self) -> None:
        result = extract_pdf((KB / "atlas-timeline.pdf").read_bytes(), max_chars=200_000)
        assert result.status == "ok"
        assert [block.page for block in result.blocks] == [1, 2]
        assert "27 November 2026: go or no go decision." in result.blocks[0].text
        assert "31 May 2027: the pilot ends." in result.blocks[1].text

    @pytest.mark.parametrize(
        ("name", "content_type", "needle"),
        [
            ("atlas-status.md", MARKDOWN, "no go live in Lyon"),
            ("atlas-faq.md", MARKDOWN, "Hanna Weber"),
            ("atlas-sites.docx", DOCX, "Trento | 60"),
            ("atlas-timeline.pdf", PDF, "15 December 2026"),
        ],
    )
    async def test_every_file_through_extract_text(self, name, content_type, needle) -> None:
        """pdf and docx go through the limited child process."""
        result = await extract_text(content_type, (KB / name).read_bytes(), max_chars=200_000)
        assert result.status == "ok", result.error
        assert needle in "\n".join(block.text for block in result.blocks)
        assert result.chars == sum(len(block.text) for block in result.blocks)


class TestPdfEdges:
    def test_pdf_without_a_text_layer_is_empty(self) -> None:
        result = extract_pdf(blank_pdf(2), max_chars=1000)
        assert result.status == "empty" and result.blocks == []

    def test_the_real_page_cap_is_two_hundred(self) -> None:
        assert MAX_PDF_PAGES == 200
        result = extract_pdf(text_pdf([[f"p{n}"] for n in range(1, MAX_PDF_PAGES + 4)]), max_chars=10_000)
        assert result.status == "truncated"
        assert [block.page for block in result.blocks][-1] == MAX_PDF_PAGES

    def test_page_cap_with_text(self, monkeypatch) -> None:
        monkeypatch.setattr(text_extract, "MAX_PDF_PAGES", 2)
        result = extract_pdf(text_pdf([[f"page number {n}"] for n in range(1, 5)]), max_chars=1000)
        assert result.status == "truncated"
        assert [block.page for block in result.blocks] == [1, 2]

    def test_encrypted_pdf_is_failed(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.encrypt("secret", algorithm="RC4-128")
        buffer = io.BytesIO()
        writer.write(buffer)
        result = extract_pdf(buffer.getvalue(), max_chars=1000)
        assert result.status == "failed"
        assert "password" in result.error


class TestDocxArchiveCheck:
    def test_a_minimal_docx_passes(self) -> None:
        check_docx_archive(minimal_docx())

    def test_not_a_zip(self) -> None:
        with pytest.raises(DocxArchiveError):
            check_docx_archive(b"PK\x03\x04 and then nothing useful")
        with pytest.raises(DocxArchiveError):
            check_docx_archive(b"%PDF-1.4")

    def test_a_zip_without_the_docx_parts(self) -> None:
        with pytest.raises(DocxArchiveError, match="missing"):
            check_docx_archive(make_zip({"hello.txt": b"hi"}))
        with pytest.raises(DocxArchiveError, match="missing"):
            check_docx_archive(make_zip({"[Content_Types].xml": b"<Types/>", "xl/workbook.xml": b"<x/>"}))

    def test_too_many_entries_is_refused_from_the_end_record(self) -> None:
        archive = minimal_docx({f"junk/{n}": b"" for n in range(MAX_ZIP_ENTRIES)})
        with pytest.raises(DocxArchiveError, match=r"\d+ entries"):
            check_docx_archive(archive)

    def test_exactly_the_entry_cap_is_allowed(self) -> None:
        check_docx_archive(minimal_docx({f"junk/{n}": b"" for n in range(MAX_ZIP_ENTRIES - 2)}))

    def test_total_uncompressed_size_over_the_cap(self) -> None:
        # Fifty-one stored entries of 1 MiB each: ratio 1 everywhere, 51 MiB in sum.
        entries = {"[Content_Types].xml": b"<Types/>", "word/document.xml": DOCX_BODY}
        entries.update({f"word/media/{n}.bin": bytes(1024 * 1024) for n in range(51)})
        archive = make_zip(entries, compress=zipfile.ZIP_STORED)
        with pytest.raises(DocxArchiveError, match="size over the cap"):
            check_docx_archive(archive)

    def test_one_entry_with_a_bomb_ratio(self) -> None:
        archive = minimal_docx({"word/media/bomb.bin": b"\x00" * (20 * 1024 * 1024)})
        with pytest.raises(DocxArchiveError, match="ratio"):
            check_docx_archive(archive)

    def test_a_zip_bomb_above_the_total_cap(self) -> None:
        archive = minimal_docx({"word/media/bomb.bin": b"\x00" * (60 * 1024 * 1024)})
        assert len(archive) < 100_000
        with pytest.raises(DocxArchiveError):
            check_docx_archive(archive)

    def test_a_lying_end_record_count_does_not_help(self) -> None:
        archive = bytearray(minimal_docx())
        end = archive.rfind(b"PK\x05\x06")
        struct.pack_into("<H", archive, end + 10, 60000)
        with pytest.raises(DocxArchiveError, match="entries"):
            check_docx_archive(bytes(archive))

    def test_small_repetitive_parts_are_not_ratio_checked(self) -> None:
        check_docx_archive(minimal_docx({"word/small.xml": b" " * 20_000}))


class TestDocxExtractionFailures:
    def test_bomb_is_refused_before_python_docx_opens_it(self, monkeypatch) -> None:
        opened = []
        import docx

        monkeypatch.setattr(docx, "Document", lambda *a, **k: opened.append(1))
        archive = minimal_docx({"word/media/bomb.bin": b"\x00" * (20 * 1024 * 1024)})
        with pytest.raises(DocxArchiveError):
            extract_docx(archive, max_chars=1000)
        assert opened == []

    async def test_bomb_through_the_child_process_is_failed(self) -> None:
        archive = minimal_docx({"word/media/bomb.bin": b"\x00" * (60 * 1024 * 1024)})
        result = await extract_text(DOCX, archive, max_chars=1000)
        assert result.status == "failed"
        assert "refused" in result.error

    async def test_a_directory_that_lies_about_sizes_fails_the_read(self) -> None:
        """Declared size 100 bytes, real size 30 MB: zipfile reads at most the declared size and the
        CRC then fails. The pre check passes, the parse fails, the file is stored as failed."""
        archive = bytearray(
            make_zip(
                {
                    "[Content_Types].xml": b"<Types/>",
                    "word/document.xml": b"<w:document>" + b"A" * (30 * 1024 * 1024),
                }
            )
        )
        central = archive.rfind(b"PK\x01\x02", 0, archive.rfind(b"PK\x05\x06"))
        assert central > 0
        struct.pack_into("<I", archive, central + 24, 100)
        result = await extract_text(DOCX, bytes(archive), max_chars=1000)
        assert result.status == "failed"

    async def test_text_that_is_not_xml_is_failed_not_raised(self) -> None:
        archive = minimal_docx({"word/document.xml": b"not xml at all"})
        result = await extract_text(DOCX, archive, max_chars=1000)
        assert result.status == "failed"
        assert result.error == "The docx could not be read"

    async def test_garbage_pdf_is_failed(self) -> None:
        result = await extract_text(PDF, b"%PDF-1.4\nthis is not a pdf\n%%EOF", max_chars=1000)
        assert result.status == "failed"
        assert result.error == "The pdf could not be read"

    async def test_cap_applies_inside_the_child(self) -> None:
        lines = [f"Line {n} of text that fills the page with words" for n in range(60)]
        result = await extract_text(PDF, text_pdf([lines]), max_chars=300)
        assert result.status == "truncated"
        assert result.chars <= 300


# The limited process ------------------------------------------------------------------------------

PROBE = textwrap.dedent(
    """
    import importlib.util, json, os, sys, time
    spec = importlib.util.spec_from_file_location("te", {module!r})
    te = importlib.util.module_from_spec(spec)
    sys.modules["te"] = te
    spec.loader.exec_module(te)
    mode, as_bytes, cpu = sys.argv[1], int(sys.argv[3]), int(sys.argv[4])
    limited = te._apply_limits(as_bytes, cpu)
    sys.stdin.buffer.read()
    def reply(text="ok", status="ok"):
        blocks = [{{"text": text, "page": None, "section": None, "atomic": False}}]
        print(json.dumps({{"result": {{"status": status, "blocks": blocks, "chars": len(text), "error": None}},
                          "memory_limited": limited, "exception": ""}}))
    if mode == "spin":
        while True:
            pass
    elif mode == "sleep":
        time.sleep(60)
    elif mode == "alloc":
        hold = bytearray(1024 * 1024 * 1024)
        reply("allocated")
    elif mode == "write":
        with open(os.path.join({tmp!r}, "out.bin"), "wb") as handle:
            handle.write(b"x" * 100000)
            handle.flush()
        reply("wrote")
    elif mode == "env":
        reply(repr(sorted(k for k in os.environ if "KEY" in k or "SECRET" in k or "S3" in k)))
    elif mode == "garbage":
        print("this is not json")
    elif mode == "huge":
        sys.stdout.write("x" * (9 * 1024 * 1024))
    elif mode == "badresult":
        print(json.dumps({{"result": {{"status": "ok", "blocks": [{{"text": "x" * 500, "page": None, "section": None, "atomic": False}}], "chars": 500, "error": None}}}}))
    elif mode == "noresult":
        sys.exit(3)
    """
)


@pytest.fixture
def probe(tmp_path: Path):
    script = tmp_path / "probe.py"
    script.write_text(PROBE.format(module=str(Path(text_extract.__file__)), tmp=str(tmp_path)))

    def run(mode: str, **kwargs):
        kwargs.setdefault("max_chars", 1000)
        kwargs.setdefault("wall_seconds", 20)
        return run_isolated(mode, b"data", script=script, **kwargs)

    return run


class TestLimitedProcess:
    def test_cpu_limit_kills_a_spinning_child(self, probe) -> None:
        started = time.monotonic()
        result = probe("spin", cpu_seconds=1, wall_seconds=20)
        assert result.status == "failed"
        assert result.error == "Extraction exceeded its limits"
        assert time.monotonic() - started < 15

    def test_wall_clock_limit_kills_a_sleeping_child(self, probe) -> None:
        started = time.monotonic()
        result = probe("sleep", wall_seconds=1)
        assert (result.status, result.error) == ("failed", "Extraction took too long")
        assert time.monotonic() - started < 10

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="macOS refuses RLIMIT_AS")
    def test_memory_limit_stops_a_child_that_allocates_a_gigabyte(self, probe) -> None:
        result = probe("alloc", as_bytes=256 * 1024 * 1024)
        assert (result.status, result.error) == ("failed", "Extraction exceeded its limits")

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="macOS refuses RLIMIT_AS")
    def test_the_same_allocation_fits_under_a_larger_limit(self, probe) -> None:
        result = probe("alloc", as_bytes=3 * 1024 * 1024 * 1024)
        assert result.status == "ok"
        assert result.blocks[0].text == "allocated"

    def test_the_child_cannot_write_files(self, probe, tmp_path: Path) -> None:
        result = probe("write")
        assert result.status == "failed"
        assert not (tmp_path / "out.bin").exists() or (tmp_path / "out.bin").stat().st_size == 0

    def test_the_child_does_not_inherit_the_environment(self, probe, monkeypatch) -> None:
        monkeypatch.setenv("JWT_SECRET_KEY", "must-not-leak")
        monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "must-not-leak-either")
        result = probe("env")
        assert result.status == "ok"
        assert result.blocks[0].text == "[]"

    @pytest.mark.parametrize(
        ("mode", "error"),
        [
            ("garbage", "Extraction failed"),
            ("huge", "Extraction exceeded its limits"),
            ("noresult", "Extraction exceeded its limits"),
        ],
    )
    def test_a_child_that_does_not_answer_with_json_is_failed(self, probe, mode, error) -> None:
        result = probe(mode)
        assert (result.status, result.error) == ("failed", error)

    def test_a_result_over_the_cap_is_refused(self, probe) -> None:
        result = probe("badresult", max_chars=100)
        assert result.status == "failed"

    def test_one_extraction_at_a_time(self, probe, monkeypatch) -> None:
        assert text_extract._SLOT.acquire(timeout=1)
        try:
            result = run_isolated("pdf", b"x", max_chars=10, wall_seconds=0.2)
        finally:
            text_extract._SLOT.release()
        assert (result.status, result.error) == ("failed", "The extractor is busy")


class TestKinds:
    async def test_unsupported_type(self) -> None:
        result = await extract_text("image/png", b"x", max_chars=10)
        assert result.status == "failed"

    async def test_txt_and_md_never_start_a_process(self, monkeypatch) -> None:
        def boom(*args, **kwargs):
            raise AssertionError("started a process")

        monkeypatch.setattr(text_extract.subprocess, "Popen", boom)
        assert (await extract_text(TEXT, b"hello", max_chars=10)).status == "ok"
        assert (await extract_text(MARKDOWN, b"# a\nb", max_chars=10)).status == "ok"
