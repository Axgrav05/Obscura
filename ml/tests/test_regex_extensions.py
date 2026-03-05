"""
Tests for expanded regex PII/PHI patterns: DOB, CREDIT_CARD, IPV4, PASSPORT.

Covers:
  - Valid format detection for all four new entity types
  - Boundary rejection (strict month/day, octet limits, casing)
  - Word-boundary guards preventing substring matches
  - Score assignments

These tests run fast (<100ms total) since they don't load any ML models.
"""

import pytest

from ml.regex_detector import RegexDetector


@pytest.fixture
def detector() -> RegexDetector:
    """Fresh RegexDetector with default settings."""
    return RegexDetector()


# ---------------------------------------------------------------------------
# DOB detection
# ---------------------------------------------------------------------------


class TestDOB:
    """Tests for date patterns in MM/DD/YYYY and YYYY-MM-DD formats."""

    def test_mm_dd_yyyy(self, detector: RegexDetector) -> None:
        """Standard MM/DD/YYYY date is detected."""
        text = "DOB: 03/15/1990 on record."
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 1
        assert dob[0].text == "03/15/1990"

    def test_yyyy_mm_dd(self, detector: RegexDetector) -> None:
        """ISO YYYY-MM-DD date is detected."""
        text = "Born on 1990-03-15 in Chicago."
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 1
        assert dob[0].text == "1990-03-15"

    def test_invalid_month_13(self, detector: RegexDetector) -> None:
        """Month 13 is rejected."""
        text = "Date: 13/40/1990"
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 0

    def test_invalid_month_00(self, detector: RegexDetector) -> None:
        """Month 00 is rejected."""
        text = "Date: 00/15/2000"
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 0

    def test_invalid_day_00(self, detector: RegexDetector) -> None:
        """Day 00 is rejected."""
        text = "Date: 01/00/2000"
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 0

    def test_invalid_day_32(self, detector: RegexDetector) -> None:
        """Day 32 is rejected."""
        text = "Date: 01/32/2000"
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 0

    def test_invalid_year_1899(self, detector: RegexDetector) -> None:
        """Year before 1900 is rejected."""
        text = "Date: 01/15/1899"
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 0

    def test_boundary_rejection(self, detector: RegexDetector) -> None:
        """Date embedded in alphanumeric string is rejected."""
        text = "Code X03/15/1990Y"
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 0

    def test_score_assignment(self, detector: RegexDetector) -> None:
        """DOB entities get the configured score."""
        text = "DOB 01/01/2000"
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 1
        assert dob[0].score == 0.95
        assert dob[0].source == "regex"

    def test_iso_invalid_month(self, detector: RegexDetector) -> None:
        """ISO format with month 13 is rejected."""
        text = "Date: 2000-13-01"
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 0


# ---------------------------------------------------------------------------
# Credit Card detection
# ---------------------------------------------------------------------------


class TestCreditCard:
    """Tests for 16-digit credit card number detection."""

    def test_dashed_format(self, detector: RegexDetector) -> None:
        """Standard dashed credit card is detected."""
        text = "Card: 1234-5678-9012-3456"
        entities = detector.detect(text)
        cc = [e for e in entities if e.entity_type == "CREDIT_CARD"]
        assert len(cc) == 1
        assert cc[0].text == "1234-5678-9012-3456"

    def test_spaced_format(self, detector: RegexDetector) -> None:
        """Space-separated credit card is detected."""
        text = "Card: 1234 5678 9012 3456"
        entities = detector.detect(text)
        cc = [e for e in entities if e.entity_type == "CREDIT_CARD"]
        assert len(cc) == 1
        assert cc[0].text == "1234 5678 9012 3456"

    def test_undashed_format(self, detector: RegexDetector) -> None:
        """Plain 16-digit credit card is detected."""
        text = "Card: 1234567890123456"
        entities = detector.detect(text)
        cc = [e for e in entities if e.entity_type == "CREDIT_CARD"]
        assert len(cc) == 1
        assert cc[0].text == "1234567890123456"

    def test_no_overlap_with_ssn(self, detector: RegexDetector) -> None:
        """9-digit SSN does not trigger credit card detection."""
        text = "SSN: 123-45-6789"
        entities = detector.detect(text)
        cc = [e for e in entities if e.entity_type == "CREDIT_CARD"]
        assert len(cc) == 0

    def test_boundary_rejection(self, detector: RegexDetector) -> None:
        """CC embedded in longer alphanumeric string is rejected."""
        text = "Code A1234-5678-9012-3456B"
        entities = detector.detect(text)
        cc = [e for e in entities if e.entity_type == "CREDIT_CARD"]
        assert len(cc) == 0

    def test_score_assignment(self, detector: RegexDetector) -> None:
        """Credit card entities get the configured score."""
        text = "Pay with 1234-5678-9012-3456"
        entities = detector.detect(text)
        cc = [e for e in entities if e.entity_type == "CREDIT_CARD"]
        assert len(cc) == 1
        assert cc[0].score == 0.99
        assert cc[0].source == "regex"


# ---------------------------------------------------------------------------
# IPv4 detection
# ---------------------------------------------------------------------------


class TestIPv4:
    """Tests for IPv4 address detection with octet validation."""

    def test_valid_ipv4(self, detector: RegexDetector) -> None:
        """Standard IPv4 address is detected."""
        text = "Server at 192.168.1.1 is down."
        entities = detector.detect(text)
        ipv4 = [e for e in entities if e.entity_type == "IPV4"]
        assert len(ipv4) == 1
        assert ipv4[0].text == "192.168.1.1"

    def test_max_octets(self, detector: RegexDetector) -> None:
        """255.255.255.255 is valid."""
        text = "Broadcast: 255.255.255.255"
        entities = detector.detect(text)
        ipv4 = [e for e in entities if e.entity_type == "IPV4"]
        assert len(ipv4) == 1
        assert ipv4[0].text == "255.255.255.255"

    def test_min_octets(self, detector: RegexDetector) -> None:
        """0.0.0.0 is valid."""
        text = "Bind to 0.0.0.0 for all interfaces."
        entities = detector.detect(text)
        ipv4 = [e for e in entities if e.entity_type == "IPV4"]
        assert len(ipv4) == 1
        assert ipv4[0].text == "0.0.0.0"

    def test_octet_256_rejected(self, detector: RegexDetector) -> None:
        """Octet 256 exceeds the valid range and is rejected."""
        text = "Invalid IP: 256.1.1.1"
        entities = detector.detect(text)
        ipv4 = [e for e in entities if e.entity_type == "IPV4"]
        assert len(ipv4) == 0

    def test_octet_999_rejected(self, detector: RegexDetector) -> None:
        """Three-digit octet above 255 is rejected."""
        text = "Not an IP: 192.168.999.1"
        entities = detector.detect(text)
        ipv4 = [e for e in entities if e.entity_type == "IPV4"]
        assert len(ipv4) == 0

    def test_boundary_rejection(self, detector: RegexDetector) -> None:
        """IP embedded in alphanumeric string is rejected."""
        text = "Version X192.168.1.1Y"
        entities = detector.detect(text)
        ipv4 = [e for e in entities if e.entity_type == "IPV4"]
        assert len(ipv4) == 0

    def test_score_assignment(self, detector: RegexDetector) -> None:
        """IPv4 entities get the configured score."""
        text = "Host: 10.0.0.1"
        entities = detector.detect(text)
        ipv4 = [e for e in entities if e.entity_type == "IPV4"]
        assert len(ipv4) == 1
        assert ipv4[0].score == 0.95
        assert ipv4[0].source == "regex"


# ---------------------------------------------------------------------------
# US Passport detection
# ---------------------------------------------------------------------------


class TestPassport:
    """Tests for US passport number detection (1 letter + 8 digits)."""

    def test_valid_passport(self, detector: RegexDetector) -> None:
        """Standard US passport number is detected."""
        text = "Passport: C12345678"
        entities = detector.detect(text)
        pp = [e for e in entities if e.entity_type == "PASSPORT"]
        assert len(pp) == 1
        assert pp[0].text == "C12345678"

    def test_lowercase_rejected(self, detector: RegexDetector) -> None:
        """Lowercase letter prefix is rejected (case-sensitive)."""
        text = "Code: a12345678"
        entities = detector.detect(text)
        pp = [e for e in entities if e.entity_type == "PASSPORT"]
        assert len(pp) == 0

    def test_too_few_digits_rejected(self, detector: RegexDetector) -> None:
        """Letter + 7 digits (too short) is not matched."""
        text = "Code: C1234567"
        entities = detector.detect(text)
        pp = [e for e in entities if e.entity_type == "PASSPORT"]
        assert len(pp) == 0

    def test_too_many_digits_rejected(self, detector: RegexDetector) -> None:
        """Letter + 9 digits is rejected by the word boundary."""
        text = "Code: C123456789"
        entities = detector.detect(text)
        pp = [e for e in entities if e.entity_type == "PASSPORT"]
        assert len(pp) == 0

    def test_boundary_rejection(self, detector: RegexDetector) -> None:
        """Passport embedded in longer word is rejected."""
        text = "ID XC12345678Y"
        entities = detector.detect(text)
        pp = [e for e in entities if e.entity_type == "PASSPORT"]
        assert len(pp) == 0

    def test_score_assignment(self, detector: RegexDetector) -> None:
        """Passport entities get the configured score."""
        text = "US Passport A98765432"
        entities = detector.detect(text)
        pp = [e for e in entities if e.entity_type == "PASSPORT"]
        assert len(pp) == 1
        assert pp[0].score == 0.99
        assert pp[0].source == "regex"
