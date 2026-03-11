"""
Tests for PIIEngine chunking, long-text stress, and stride deduplication.

Tests _chunk_text() as pure unit tests (no mock needed).
Tests _detect_bert() and redact() with mocked pipeline to avoid model downloads.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ml.pii_engine import PIIEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> PIIEngine:
    """PIIEngine with mocked BERT pipeline — no model download in CI."""
    eng = PIIEngine(model_id="mock", enable_regex=True)
    eng._pipeline = MagicMock(return_value=[])
    return eng


@pytest.fixture
def engine_no_regex() -> PIIEngine:
    """PIIEngine with regex disabled — isolates BERT path."""
    eng = PIIEngine(model_id="mock", enable_regex=False)
    eng._pipeline = MagicMock(return_value=[])
    return eng


def _make_entity(
    text: str,
    entity_type: str,
    start: int,
    end: int,
    score: float = 0.99,
    source: str = "bert",
) -> dict:
    """Build a raw pipeline dict mimicking HuggingFace NER output."""
    return {
        "word": text,
        "entity_group": entity_type,
        "score": score,
        "start": start,
        "end": end,
    }


# ---------------------------------------------------------------------------
# TestChunkText — pure unit tests, no mock needed
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_short_text_returns_single_chunk(self, engine: PIIEngine) -> None:
        text = "Hello world. My name is John Smith."
        chunks = engine._chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == (text, 0)

    def test_text_at_exact_chunk_size_is_single_chunk(self, engine: PIIEngine) -> None:
        # Text exactly at chunk_size boundary should not be split.
        text = "A" * engine.chunk_size
        chunks = engine._chunk_text(text)
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self, engine: PIIEngine) -> None:
        # Build a text clearly exceeding chunk_size with sentence delimiters.
        sentence = "This is a sample sentence with enough words to matter. "
        text = sentence * 60  # ~3300 chars, well over default 1500
        chunks = engine._chunk_text(text)
        assert len(chunks) >= 2

    def test_global_offsets_reconstruct_original(self, engine: PIIEngine) -> None:
        # Each chunk must be an exact substring at its stated global offset.
        sentence = "Word " * 50 + ". "
        text = sentence * 20
        chunks = engine._chunk_text(text)
        for chunk_text, offset in chunks:
            assert text[offset : offset + len(chunk_text)] == chunk_text

    def test_stride_overlap_last_sentence_repeats(self, engine: PIIEngine) -> None:
        # The last sentence of chunk N must appear at the start of chunk N+1.
        sentence = "Alpha beta gamma delta epsilon zeta. "
        text = sentence * 80  # ~2960 chars — forces chunking at chunk_size=1500
        chunks = engine._chunk_text(text)
        assert len(chunks) >= 2
        chunk0_text, chunk0_offset = chunks[0]
        chunk1_text, chunk1_offset = chunks[1]
        # Chunk 1 starts before chunk 0 ends — stride overlap exists.
        assert chunk1_offset < chunk0_offset + len(chunk0_text)
        # The overlapping region is identical in both chunks.
        overlap_start = chunk1_offset
        overlap_end = chunk0_offset + len(chunk0_text)
        chunk0_tail = chunk0_text[overlap_start - chunk0_offset :]
        chunk1_head = chunk1_text[: overlap_end - chunk1_offset]
        assert chunk0_tail == chunk1_head

    def test_single_oversized_sentence_no_infinite_loop(
        self, engine: PIIEngine
    ) -> None:
        # A sentence longer than chunk_size must still be returned as one chunk.
        text = "A" * (engine.chunk_size + 500)  # 2000 chars, no delimiters
        chunks = engine._chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == (text, 0)

    def test_chunk_offsets_are_monotonically_increasing(
        self, engine: PIIEngine
    ) -> None:
        sentence = "The patient presented with no prior history of conditions. "
        text = sentence * 60
        chunks = engine._chunk_text(text)
        offsets = [o for _, o in chunks]
        assert offsets == sorted(offsets)
        assert offsets[0] == 0


# ---------------------------------------------------------------------------
# TestLongTextStressTest — regex-detected SSNs in a 3000-word payload
# ---------------------------------------------------------------------------


class TestLongTextStressTest:
    """Verify chunking doesn't break regex detection for long payloads.

    SSNs are detected by the regex layer (which runs on the full text, not
    chunks), so no BERT mock setup is needed beyond suppressing BERT output.
    The test validates that redact() handles a ~3000-word text without error
    and correctly redacts all injected SSNs.
    """

    def _build_long_text(self) -> tuple[str, int, int, int]:
        """Build a ~3000-word synthetic payload with SSNs at start, middle, end.

        Returns (text, start_pos, mid_pos, end_pos).
        """
        # Filler paragraph (~150 words, ~900 chars).
        filler = (
            "The quarterly review meeting was attended by representatives from "
            "all major departments. Finance reported steady growth while operations "
            "highlighted process improvements in the supply chain. Human resources "
            "noted increased employee retention across all divisions. The board "
            "expressed confidence in the current strategic direction and approved "
            "the proposed budget for the upcoming fiscal period. Additional agenda "
            "items included infrastructure upgrades and vendor contract renewals. "
        )

        # Beginning: SSN at the very start.
        start_ssn = "123-45-6789"
        beginning = (
            f"Patient SSN: {start_ssn} was admitted on the first floor. {filler}"
        )

        # Middle: SSN buried after enough text to cross a chunk boundary.
        # Repeat filler ~8 times to push past the 1500-char chunk boundary.
        mid_prefix = filler * 8
        mid_ssn = "234-56-7890"
        middle = f"{mid_prefix}Secondary record SSN: {mid_ssn} is attached. {filler}"

        # End: SSN at the very last sentence.
        end_filler = filler * 6
        end_ssn = "345-67-8901"
        ending = f"{end_filler}Final note: SSN {end_ssn} must be redacted."

        text = beginning + middle + ending

        # Locate exact positions for assertion.
        start_pos = text.index(start_ssn)
        mid_pos = text.index(mid_ssn)
        end_pos = text.index(end_ssn)
        return text, start_pos, mid_pos, end_pos

    def test_all_three_ssns_redacted(self, engine: PIIEngine) -> None:
        text, _s, _m, _e = self._build_long_text()
        result = engine.redact(text)
        assert "123-45-6789" not in result.masked_text
        assert "234-56-7890" not in result.masked_text
        assert "345-67-8901" not in result.masked_text

    def test_three_ssn_tokens_in_mapping(self, engine: PIIEngine) -> None:
        text, _, _, _ = self._build_long_text()
        result = engine.redact(text)
        ssn_tokens = [k for k in result.mapping if k.startswith("[SSN_")]
        assert len(ssn_tokens) == 3

    def test_no_index_error_on_long_payload(self, engine: PIIEngine) -> None:
        # Regression: must not raise IndexError or similar from token overflow.
        text, _, _, _ = self._build_long_text()
        try:
            engine.redact(text)
        except Exception as exc:
            pytest.fail(f"redact() raised {type(exc).__name__} on long payload: {exc}")

    def test_restore_returns_original_for_ssns(self, engine: PIIEngine) -> None:
        text, _, _, _ = self._build_long_text()
        result = engine.redact(text)
        restored = PIIEngine.restore(result.masked_text, result.mapping)
        assert "123-45-6789" in restored
        assert "234-56-7890" in restored
        assert "345-67-8901" in restored


# ---------------------------------------------------------------------------
# TestStrideDeduplication — BERT entity detected in two overlapping chunks
# ---------------------------------------------------------------------------


class TestStrideDeduplication:
    """Verify that an entity captured in the stride overlap of two chunks
    is deduplicated to a single DetectedEntity in the final output."""

    def test_duplicate_bert_entity_collapsed_to_one(
        self, engine_no_regex: PIIEngine
    ) -> None:
        # Build a text long enough to produce at least 2 chunks.
        sentence = (
            "The clinical notes reference a complex medical history for evaluation. "
        )
        # Position "John Smith" at the overlap zone between chunk 0 and chunk 1.
        # We simulate this by mocking the pipeline's batch output directly.
        text = sentence * 40  # ~2800 chars — will chunk

        chunks = engine_no_regex._chunk_text(text)
        assert len(chunks) >= 2, "Test prerequisite: text must produce multiple chunks"

        # The overlap zone starts at chunk[1]'s offset.
        overlap_start = chunks[1][1]
        overlap_end = chunks[0][1] + len(chunks[0][0])
        assert (
            overlap_start < overlap_end
        ), "Test prerequisite: stride overlap must exist"

        # Simulate: both chunks detect the same PERSON at global offset overlap_start.
        # Local offset in chunk 0 = overlap_start - chunk0_offset.
        # Local offset in chunk 1 = overlap_start - chunk1_offset.
        chunk0_offset = chunks[0][1]
        chunk1_offset = chunks[1][1]
        local_start_in_c0 = overlap_start - chunk0_offset
        local_start_in_c1 = overlap_start - chunk1_offset

        entity_len = 10

        def mock_batch(inputs: list[str]) -> list[list[dict]]:
            results: list[list[dict]] = []
            for idx, chunk_text in enumerate(inputs):
                if idx == 0:
                    results.append(
                        [
                            _make_entity(
                                "John Smith",
                                "PER",
                                local_start_in_c0,
                                local_start_in_c0 + entity_len,
                                score=0.99,
                            )
                        ]
                    )
                elif idx == 1:
                    results.append(
                        [
                            _make_entity(
                                "John Smith",
                                "PER",
                                local_start_in_c1,
                                local_start_in_c1 + entity_len,
                                score=0.97,
                            )
                        ]
                    )
                else:
                    results.append([])
            return results

        engine_no_regex._pipeline = MagicMock(side_effect=mock_batch)

        entities = engine_no_regex._detect_bert(text)

        person_entities = [e for e in entities if e.entity_type == "PERSON"]
        assert (
            len(person_entities) == 1
        ), f"Expected 1 PERSON entity after deduplication, got {len(person_entities)}"
        assert person_entities[0].start == overlap_start
        assert person_entities[0].end == overlap_start + entity_len
