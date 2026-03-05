"""
Tests for configurable redaction via the disabled_entities parameter.

Covers:
  - Disabled entity types are left intact in the masked text
  - Disabled entities do not populate the substitution mapping dict
  - Conflict resolution is preserved even when the winning entity is disabled
    (suppressed entities must not accidentally emerge)

These tests use a real RegexDetector but mock the BERT pipeline to avoid
model downloads in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml.pii_engine import PIIEngine
from ml.schemas import DetectedEntity


@pytest.fixture
def engine() -> PIIEngine:
    """PIIEngine with a mocked BERT pipeline that returns no entities.

    The regex detector is real, so structured PII (SSN, PHONE, etc.)
    is detected normally. BERT entities can be injected per-test by
    patching _detect_bert.
    """
    eng = PIIEngine(model_id="mock", enable_regex=True)
    # Install a no-op pipeline so _ensure_loaded() doesn't raise.
    eng._pipeline = MagicMock()
    return eng


# ---------------------------------------------------------------------------
# Basic disabled_entities behavior
# ---------------------------------------------------------------------------


class TestDisabledEntitiesBasic:
    """Core tests for the disabled_entities filtering logic."""

    def test_disabled_ssn_left_intact(self, engine: PIIEngine) -> None:
        """An SSN is left untouched when SSN is disabled."""
        text = "SSN: 123-45-6789"
        with patch.object(engine, "_detect_bert", return_value=[]):
            result = engine.redact(text, disabled_entities=["SSN"])
        assert result.masked_text == text
        assert len(result.entities) == 0
        assert len(result.mapping) == 0

    def test_disabled_ssn_no_mapping_entry(self, engine: PIIEngine) -> None:
        """Disabled entities produce no mapping tokens."""
        text = "SSN is 123-45-6789 and phone is (555) 123-4567"
        with patch.object(engine, "_detect_bert", return_value=[]):
            result = engine.redact(text, disabled_entities=["SSN"])
        # SSN should be left intact, phone should be masked.
        assert "123-45-6789" in result.masked_text
        assert "[PHONE_1]" in result.masked_text
        assert "[SSN_1]" not in result.masked_text
        # Only phone in mapping.
        assert len(result.mapping) == 1
        assert "[PHONE_1]" in result.mapping

    def test_disabled_multiple_types(self, engine: PIIEngine) -> None:
        """Multiple entity types can be disabled at once."""
        text = "SSN 123-45-6789, phone (555) 123-4567, email test@example.com"
        with patch.object(engine, "_detect_bert", return_value=[]):
            result = engine.redact(text, disabled_entities=["SSN", "PHONE"])
        # SSN and phone left intact, email masked.
        assert "123-45-6789" in result.masked_text
        assert "(555) 123-4567" in result.masked_text
        assert "[EMAIL_1]" in result.masked_text

    def test_none_disabled_masks_all(self, engine: PIIEngine) -> None:
        """When disabled_entities is None, all entities are masked (default)."""
        text = "SSN: 123-45-6789"
        with patch.object(engine, "_detect_bert", return_value=[]):
            result = engine.redact(text, disabled_entities=None)
        assert "[SSN_1]" in result.masked_text
        assert "123-45-6789" not in result.masked_text

    def test_empty_list_masks_all(self, engine: PIIEngine) -> None:
        """An empty disabled list is falsy, so all entities are masked."""
        text = "SSN: 123-45-6789"
        with patch.object(engine, "_detect_bert", return_value=[]):
            result = engine.redact(text, disabled_entities=[])
        assert "[SSN_1]" in result.masked_text

    def test_no_entities_returns_original(self, engine: PIIEngine) -> None:
        """Text with no PII returns unchanged even with disabled list."""
        text = "No PII here."
        with patch.object(engine, "_detect_bert", return_value=[]):
            result = engine.redact(text, disabled_entities=["SSN"])
        assert result.masked_text == text
        assert len(result.mapping) == 0


# ---------------------------------------------------------------------------
# Conflict resolution + disabled_entities interaction
# ---------------------------------------------------------------------------


class TestDisabledEntitiesConflictResolution:
    """Ensure disabled_entities doesn't break conflict resolution.

    The key invariant: filtering happens AFTER detect() completes
    (including merge_entities). So even if the winning entity is
    disabled, the suppressed loser must NOT re-emerge.
    """

    def test_disabled_winner_does_not_expose_loser(self, engine: PIIEngine) -> None:
        """If regex SSN wins an exact overlap and SSN is disabled,
        the losing BERT entity must not appear in the output.

        Scenario: BERT detects "123-45-6789" as PERSON (score 0.92).
        Regex detects the same span as SSN (score 0.99). SSN wins
        exact-overlap resolution. Then SSN is disabled — the PERSON
        entity was already eliminated by conflict resolution and must
        not re-emerge.
        """
        text = "Patient ID: 123-45-6789"

        # Simulate BERT detecting the SSN span as PERSON.
        fake_bert = [
            DetectedEntity(
                text="123-45-6789",
                entity_type="PERSON",
                start=12,
                end=23,
                score=0.92,
                token="",
                source="bert",
            )
        ]
        with patch.object(engine, "_detect_bert", return_value=fake_bert):
            result = engine.redact(text, disabled_entities=["SSN"])

        # SSN is disabled so it's filtered out. But the PERSON entity
        # was already eliminated during merge_entities (SSN won).
        # Neither should appear in the masked text.
        assert "123-45-6789" in result.masked_text
        assert "[PERSON_1]" not in result.masked_text
        assert "[SSN_1]" not in result.masked_text
        assert len(result.mapping) == 0
