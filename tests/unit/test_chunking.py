"""Chunking: about 150 tokens, one page and one section per chunk, table rows kept whole."""
import time
from pathlib import Path

import pytest

from utils.chunking import CHARS_PER_TOKEN, chunk_blocks, estimate_tokens
from utils.text_extract import DOCX, MARKDOWN, PDF, Block, extract_text

KB = Path(__file__).resolve().parents[1] / "data" / "agentar_kb"
BUDGET = 150 * CHARS_PER_TOKEN


def texts(chunks):
    return [chunk.text for chunk in chunks]


def test_no_blocks_no_chunks() -> None:
    assert chunk_blocks([]) == []


def test_estimate_tokens_is_four_characters_per_token() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_small_blocks_pack_into_one_chunk_in_order() -> None:
    chunks = chunk_blocks([Block("first fact."), Block("second fact."), Block("third fact.")])
    assert texts(chunks) == ["first fact.\nsecond fact.\nthird fact."]
    assert (chunks[0].n, chunks[0].page) == (0, None)


def test_a_chunk_stays_within_the_token_budget_and_numbers_run_on() -> None:
    blocks = [Block(f"Paragraph {n} " + "word " * 30) for n in range(40)]
    chunks = chunk_blocks(blocks)
    assert len(chunks) > 3
    assert [chunk.n for chunk in chunks] == list(range(len(chunks)))
    assert all(estimate_tokens(chunk.text) <= 150 for chunk in chunks)
    # about 150 tokens: the packing is greedy, so chunks are close to the budget
    assert sum(estimate_tokens(chunk.text) for chunk in chunks[:-1]) / (len(chunks) - 1) > 100
    assert "".join(texts(chunks)).count("Paragraph") == 40


def test_target_tokens_is_a_parameter() -> None:
    blocks = [Block("x" * 90) for _ in range(10)]
    assert len(chunk_blocks(blocks, target_tokens=30)) == 10  # 120 characters: one block per chunk
    assert len(chunk_blocks(blocks, target_tokens=1000)) == 1


def test_a_page_change_starts_a_new_chunk_and_keeps_the_page() -> None:
    chunks = chunk_blocks([Block("on one", page=1), Block("also one", page=1), Block("on two", page=2)])
    assert [(chunk.page, chunk.text) for chunk in chunks] == [(1, "on one\nalso one"), (2, "on two")]


def test_a_section_change_starts_a_new_chunk_and_the_heading_leads_every_chunk() -> None:
    blocks = [
        Block("Trento seats are 60.", section="Sites"),
        Block("Giulia leads Trento.", section="Contacts"),
        Block("Bonn room is Rhine 2.", section="Contacts"),
    ]
    assert texts(chunk_blocks(blocks)) == [
        "Sites\nTrento seats are 60.",
        "Contacts\nGiulia leads Trento.\nBonn room is Rhine 2.",
    ]


def test_a_long_section_repeats_its_heading_in_each_chunk() -> None:
    blocks = [Block("fact " * 40, section="Long section") for _ in range(6)]
    chunks = chunk_blocks(blocks)
    assert len(chunks) > 1
    assert all(chunk.text.startswith("Long section\n") for chunk in chunks)
    assert all(len(chunk.text) <= BUDGET for chunk in chunks)


def test_a_very_long_heading_is_clipped() -> None:
    chunks = chunk_blocks([Block("body", section="H" * 500)])
    first_line = chunks[0].text.split("\n")[0]
    assert len(first_line) <= 100 and first_line.endswith("…")


def test_table_rows_are_atomic_and_stay_together() -> None:
    rows = ["Site | Pilot seats | Start", "Trento | 60 | 1 December 2026", "Bonn | 60 | 1 December 2026"]
    filler = Block("lead in " * 65, section="Sites")  # 520 characters, the last row spills over
    chunks = chunk_blocks([filler] + [Block(row, section="Sites", atomic=True) for row in rows])
    # every row is whole inside exactly one chunk
    for row in rows:
        assert sum(row in chunk.text.split("\n") for chunk in chunks) == 1
    assert len(chunks) == 2


def test_an_atomic_row_longer_than_the_room_is_still_one_line() -> None:
    row = "Site | " + "cell " * 150  # 750 characters, over the 600 budget, under four times it
    chunks = chunk_blocks([Block(row.strip(), atomic=True)])
    assert texts(chunks) == [row.strip()]


def test_an_atomic_row_over_four_times_the_budget_is_split_like_text() -> None:
    chunks = chunk_blocks([Block("cell " * 1000, atomic=True)])
    assert len(chunks) > 1
    assert all(len(chunk.text) <= BUDGET for chunk in chunks)


def test_a_long_paragraph_splits_at_lines_then_sentences_then_words() -> None:
    lines = "\n".join(f"- Fact number {n} is stated here." for n in range(60))
    chunks = chunk_blocks([Block(lines)])
    assert all(len(chunk.text) <= BUDGET for chunk in chunks)
    assert all(line.startswith("- Fact") for chunk in chunks for line in chunk.text.split("\n"))  # no cut lines

    sentences = " ".join(f"This is sentence {n} of the long paragraph." for n in range(40))
    chunks = chunk_blocks([Block(sentences)])
    assert all(len(chunk.text) <= BUDGET for chunk in chunks)
    assert all(chunk.text.rstrip().endswith(".") for chunk in chunks)

    words = " ".join(f"w{n}" for n in range(1000))
    assert all(len(chunk.text) <= BUDGET for chunk in chunk_blocks([Block(words)]))


def test_a_word_longer_than_the_budget_is_cut() -> None:
    chunks = chunk_blocks([Block("x" * 2000)])
    assert [len(chunk.text) for chunk in chunks] == [600, 600, 600, 200]


def test_no_text_is_lost() -> None:
    paragraph = " ".join(f"word{n}" for n in range(500))
    chunks = chunk_blocks([Block(paragraph)])
    assert " ".join(texts(chunks)).split() == paragraph.split()


def test_two_hundred_thousand_characters_chunk_quickly() -> None:
    blocks = [Block(("sentence of ordinary prose. " * 20).strip(), page=n // 50 + 1) for n in range(360)]
    started = time.monotonic()
    chunks = chunk_blocks(blocks)
    assert time.monotonic() - started < 1.5
    assert sum(len(chunk.text) for chunk in chunks) >= 190_000


@pytest.mark.parametrize(
    ("name", "content_type", "expected"),
    [
        (
            "atlas-status.md",
            MARKDOWN,
            [(None, "Atlas pilot, status for the steering call\n- Go live planned on 1 December 2026 at two sites, Trento and Bonn")],
        ),
        ("atlas-timeline.pdf", PDF, [(1, "Atlas pilot timeline\n14 November 2026"), (2, "15 December 2026: first usage review.")]),
        (
            "atlas-sites.docx",
            DOCX,
            [
                (None, "Atlas pilot sites\nSeats and contacts per site for the Atlas pilot.\nSite | Pilot seats | Start\nTrento | 60 | 1 December 2026"),
                (None, "Contacts and rooms\nThe Trento site lead is Giulia Conti."),
            ],
        ),
    ],
)
async def test_the_e0_files(name, content_type, expected) -> None:
    extraction = await extract_text(content_type, (KB / name).read_bytes(), max_chars=200_000)
    chunks = chunk_blocks(extraction.blocks)
    for (page, needle), chunk in zip(expected, chunks):
        assert chunk.page == page
        assert needle in chunk.text
    assert len(chunks) == len(expected)


async def test_the_faq_splits_in_two_and_each_chunk_keeps_the_title() -> None:
    extraction = await extract_text(MARKDOWN, (KB / "atlas-faq.md").read_bytes(), max_chars=200_000)
    chunks = chunk_blocks(extraction.blocks)
    assert len(chunks) == 2
    assert all(chunk.text.startswith("Atlas pilot, frequently asked questions\n- ") for chunk in chunks)
    assert "Hanna Weber" in chunks[1].text
    facts = [line for chunk in chunks for line in chunk.text.split("\n") if line.startswith("- ")]
    assert len(facts) == 10


async def test_the_docx_row_trento_60_is_in_one_line() -> None:
    extraction = await extract_text(DOCX, (KB / "atlas-sites.docx").read_bytes(), max_chars=200_000)
    lines = [line for chunk in chunk_blocks(extraction.blocks) for line in chunk.text.split("\n")]
    assert "Trento | 60 | 1 December 2026" in lines
