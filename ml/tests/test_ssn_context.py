"""
Gold-standard test suite for SSN detection with context disambiguation.

Tests the RegexDetector and merge_entities independently of the BERT model,
ensuring:
  - Dashed SSNs are always detected with high confidence
  - IRS-invalid SSNs are rejected (area 000/666/900+, group 00, serial 0000)
  - Dashless SSNs with trigger words are detected
  - Dashless 9-digit numbers in non-SSN context are rejected
  - Conflict resolution correctly handles overlapping spans

These tests run fast (<100ms total) since they don't load any ML models.
"""

import pytest

from ml.pii_engine import merge_entities
from ml.regex_detector import RegexDetector
from ml.schemas import DetectedEntity


@pytest.fixture
def detector() -> RegexDetector:
    """Fresh RegexDetector with default settings."""
    return RegexDetector()


# ---------------------------------------------------------------------------
# Dashed SSN detection
# ---------------------------------------------------------------------------


class TestDashedSSN:
    """Tests for standard dashed SSN format (XXX-XX-XXXX)."""

    def test_basic_dashed_ssn(self, detector: RegexDetector) -> None:
        """Standard dashed SSN is detected with high confidence."""
        text = "SSN on file: 123-45-6789."
        entities = detector.detect(text)
        assert len(entities) == 1
        assert entities[0].entity_type == "SSN"
        assert entities[0].text == "123-45-6789"
        assert entities[0].score == 0.99
        assert entities[0].source == "regex"

    def test_invalid_area_666(self, detector: RegexDetector) -> None:
        """Area 666 is rejected per IRS rules."""
        entities = detector.detect("SSN: 666-45-6789")
        assert len(entities) == 0

    def test_invalid_area_900(self, detector: RegexDetector) -> None:
        """Area 900-999 is rejected per IRS rules."""
        entities = detector.detect("SSN: 900-12-3456")
        assert len(entities) == 0

    def test_invalid_group_00(self, detector: RegexDetector) -> None:
        """Group 00 is rejected per IRS rules."""
        entities = detector.detect("SSN: 123-00-6789")
        assert len(entities) == 0

    def test_invalid_serial_0000(self, detector: RegexDetector) -> None:
        """Serial 0000 is rejected per IRS rules."""
        entities = detector.detect("SSN: 123-45-0000")
        assert len(entities) == 0


# ---------------------------------------------------------------------------
# Dashless SSN context scoring
# ---------------------------------------------------------------------------


class TestDashlessSSNContext:
    """Tests for dashless SSN detection with context disambiguation.

    Each test validates that trigger words boost detection and negative
    context words suppress it. The 9-digit numbers used are all
    IRS-structurally-valid so regex fires — only context decides.
    """

    def test_trigger_word_ssn(self, detector: RegexDetector) -> None:
        """'SSN' trigger word causes detection."""
        # Gold Standard #1
        text = "Applicant's SSN is 123456789, verified via W-2."
        entities = detector.detect(text)
        assert len(entities) == 1
        assert entities[0].entity_type == "SSN"
        assert entities[0].text == "123456789"
        assert entities[0].score >= 0.70

    def test_phone_context_rejected(self, detector: RegexDetector) -> None:
        """'Call' + 'phone' context suppresses SSN classification."""
        # Gold Standard #2
        text = "Call the vendor at 941555012 to confirm delivery."
        entities = detector.detect(text)
        assert len(entities) == 0

    def test_social_security_phrase(self, detector: RegexDetector) -> None:
        """'Social security number' phrase triggers high-confidence detection."""
        # Gold Standard #3
        text = "Social security number on file: 234567890."
        entities = detector.detect(text)
        assert len(entities) == 1
        assert entities[0].score >= 0.85  # base + trigger + phrase

    def test_tracking_context_rejected(self, detector: RegexDetector) -> None:
        """'Tracking number' context suppresses SSN classification."""
        # Gold Standard #4
        text = "Tracking number 345678901 shipped via FedEx."
        entities = detector.detect(text)
        assert len(entities) == 0

    def test_tax_id_trigger(self, detector: RegexDetector) -> None:
        """'Tax ID' + 'W-9' context triggers detection."""
        # Gold Standard #5
        text = "Employee tax ID 456789012 must be updated on the W-9."
        entities = detector.detect(text)
        assert len(entities) == 1
        assert entities[0].score >= 0.70

    def test_zip_context_rejected(self, detector: RegexDetector) -> None:
        """'ZIP code' + 'postal' context suppresses SSN classification."""
        # Gold Standard #6
        text = "Enter ZIP code 100234567 for extended postal lookup."
        entities = detector.detect(text)
        assert len(entities) == 0

    def test_background_check_trigger(self, detector: RegexDetector) -> None:
        """'Background check' + 'identity' context triggers detection."""
        # Gold Standard #7
        text = "Background check confirmed identity 567890123 " "for Jane Doe."
        entities = detector.detect(text)
        assert len(entities) == 1
        assert entities[0].score >= 0.70

    def test_serial_context_rejected(self, detector: RegexDetector) -> None:
        """'Serial number' context suppresses SSN classification."""
        # Gold Standard #8
        text = "Serial number 678901234 on the replacement device."
        entities = detector.detect(text)
        assert len(entities) == 0

    def test_ss_hash_trigger(self, detector: RegexDetector) -> None:
        """'SS#' abbreviation triggers detection."""
        # Gold Standard #9
        text = "Patient SS# 789012345 requires HIPAA verification."
        entities = detector.detect(text)
        assert len(entities) == 1
        assert entities[0].score >= 0.70

    def test_invoice_context_rejected(self, detector: RegexDetector) -> None:
        """'Invoice' + 'account' context suppresses SSN classification."""
        # Gold Standard #10
        text = "Invoice 890123456 was processed for account closure."
        entities = detector.detect(text)
        assert len(entities) == 0

    def test_no_context_rejected(self, detector: RegexDetector) -> None:
        """Bare 9-digit number with neutral context is not detected."""
        text = "The value is 345678901 in the dataset."
        entities = detector.detect(text)
        assert len(entities) == 0  # Score 0.40 < threshold 0.70

    def test_multiple_ssns_in_one_string(self, detector: RegexDetector) -> None:
        """Multiple dashed SSNs in a single string are all detected."""
        text = "SSN: 123-45-6789 and SSN: 234-56-7890 on file."
        entities = detector.detect(text)
        assert len(entities) == 2
        assert entities[0].text == "123-45-6789"
        assert entities[1].text == "234-56-7890"

    def test_alpha_adjacent_dashed_rejected(self, detector: RegexDetector) -> None:
        """Dashed SSN flush against letters is not a real SSN."""
        text = "CodeA123-45-6789B is not an SSN."
        entities = detector.detect(text)
        assert len(entities) == 0

    def test_alpha_adjacent_dashless_rejected(self, detector: RegexDetector) -> None:
        """Dashless 9-digit number flush against letters is rejected."""
        text = "The SSN is REF123456789X per the form."
        entities = detector.detect(text)
        assert len(entities) == 0

    def test_ssn_after_colon_accepted(self, detector: RegexDetector) -> None:
        """SSN after punctuation (colon) is still detected — colon is not \\w."""
        text = "SSN:123-45-6789"
        entities = detector.detect(text)
        assert len(entities) == 1
        assert entities[0].text == "123-45-6789"


# ---------------------------------------------------------------------------
# Merge / conflict resolution
# ---------------------------------------------------------------------------


class TestMergeEntities:
    """Tests for hybrid conflict resolution (GAMEPLAN.md 2.3 rules)."""

    def test_no_overlap_keeps_both(self) -> None:
        """Non-overlapping BERT + regex entities are both kept (Rule 4)."""
        bert = [DetectedEntity("John", "PERSON", 0, 4, 0.95, "", "bert")]
        regex = [DetectedEntity("123-45-6789", "SSN", 20, 31, 0.99, "", "regex")]
        merged = merge_entities(bert, regex)
        assert len(merged) == 2
        assert merged[0].entity_type == "PERSON"
        assert merged[1].entity_type == "SSN"

    def test_exact_overlap_regex_wins_structured(self) -> None:
        """On exact overlap, regex wins for structured type SSN (Rule 1)."""
        bert = [DetectedEntity("123456789", "MISC", 10, 19, 0.90, "", "bert")]
        regex = [DetectedEntity("123456789", "SSN", 10, 19, 0.75, "", "regex")]
        merged = merge_entities(bert, regex)
        assert len(merged) == 1
        assert merged[0].entity_type == "SSN"
        assert merged[0].source == "regex"

    def test_exact_overlap_bert_wins_semantic(self) -> None:
        """On exact overlap, BERT wins for semantic types (Rule 1)."""
        bert = [DetectedEntity("Acme Corp", "ORGANIZATION", 5, 14, 0.95, "", "bert")]
        regex = [DetectedEntity("Acme Corp", "MISC", 5, 14, 0.80, "", "regex")]
        merged = merge_entities(bert, regex)
        assert len(merged) == 1
        assert merged[0].entity_type == "ORGANIZATION"

    def test_nested_keeps_outer(self) -> None:
        """On nested spans, outer (longer) span wins (Rule 3)."""
        outer = DetectedEntity("John Smith Jr", "PERSON", 0, 13, 0.92, "", "bert")
        inner = DetectedEntity("John Smith", "PERSON", 0, 10, 0.95, "", "bert")
        merged = merge_entities([outer, inner], [])
        assert len(merged) == 1
        assert merged[0].text == "John Smith Jr"

    def test_three_way_overlap_long_span_wins(self) -> None:
        """A long span overlapping two shorter accepted spans replaces both."""
        # "John Smith Jr from Acme" as one long PERSON span overlaps
        # with both "John Smith" (PERSON) and "Acme" (ORG).
        short_a = DetectedEntity("John Smith", "PERSON", 0, 10, 0.95, "", "bert")
        short_b = DetectedEntity("Acme", "ORG", 19, 23, 0.90, "", "bert")
        long_span = DetectedEntity(
            "John Smith Jr from Acme", "PERSON", 0, 23, 0.88, "", "bert"
        )
        merged = merge_entities([short_a, short_b, long_span], [])
        # Long span is longer than both shorts, so it should win and
        # replace both, yielding exactly one entity.
        assert len(merged) == 1
        assert merged[0].text == "John Smith Jr from Acme"

    def test_three_way_overlap_short_wins(self) -> None:
        """If a candidate is shorter than an accepted entity, accepted survives."""
        long_span = DetectedEntity(
            "John Smith Jr from Acme", "PERSON", 0, 23, 0.88, "", "bert"
        )
        short_a = DetectedEntity("John", "PERSON", 0, 4, 0.95, "", "bert")
        # long_span is sorted first (earlier start, longer length), then short_a.
        # short_a overlaps long_span but is shorter, so long_span wins.
        merged = merge_entities([long_span, short_a], [])
        assert len(merged) == 1
        assert merged[0].text == "John Smith Jr from Acme"

    def test_empty_inputs(self) -> None:
        """Empty inputs return empty list."""
        assert merge_entities([], []) == []
