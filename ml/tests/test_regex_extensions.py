"""
Tests for expanded regex PII/PHI patterns: DOB, CREDIT_CARD, IPV4, PASSPORT.

Covers:
  - Valid format detection for all four new entity types
  - Boundary rejection (strict month/day, octet limits, casing)
  - Word-boundary guards preventing substring matches
  - Score assignments
  - DOB context-aware scoring (trigger/negative word disambiguation)
  - Credit card undashed context-aware scoring

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
        """DOB entities with trigger context get a high score."""
        text = "DOB 01/01/2000"
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 1
        assert dob[0].score >= 0.90
        assert dob[0].source == "regex"

    def test_iso_invalid_month(self, detector: RegexDetector) -> None:
        """ISO format with month 13 is rejected."""
        text = "Date: 2000-13-01"
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 0


class TestDOBContext:
    """Tests for DOB context-aware scoring."""

    def test_trigger_word_dob(self, detector: RegexDetector) -> None:
        """'DOB' trigger word causes detection with high score."""
        text = "Patient DOB: 01/15/1990 on file."
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 1
        assert dob[0].score >= 0.90

    def test_trigger_word_born(self, detector: RegexDetector) -> None:
        """'born' trigger word causes detection."""
        text = "She was born 1990-01-15 in Boston."
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 1
        assert dob[0].score >= 0.90

    def test_trigger_word_birthday(self, detector: RegexDetector) -> None:
        """'birthday' trigger word causes detection."""
        text = "His birthday is 03/15/1990 per records."
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 1
        assert dob[0].score >= 0.90

    def test_trigger_phrase_date_of_birth(self, detector: RegexDetector) -> None:
        """'date of birth' phrase triggers high-confidence detection."""
        text = "Date of birth: 07/04/1985 confirmed."
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 1
        assert dob[0].score >= 0.90

    def test_negative_meeting_context(self, detector: RegexDetector) -> None:
        """'meeting' + 'scheduled' context suppresses DOB detection."""
        text = "Meeting scheduled for 01/15/2026 in the conference room."
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 0

    def test_negative_invoice_context(self, detector: RegexDetector) -> None:
        """'invoice' context suppresses DOB detection."""
        text = "Invoice dated 03/15/2025 was filed."
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 0

    def test_neutral_context_suppressed(self, detector: RegexDetector) -> None:
        """Date with neutral context is suppressed (below threshold)."""
        text = "The value 01/15/1990 was noted in the dataset."
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 0

    def test_iso_format_with_trigger(self, detector: RegexDetector) -> None:
        """ISO format date with trigger word is detected."""
        text = "Born on 1990-01-15 in Chicago."
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        assert len(dob) == 1
        assert dob[0].text == "1990-01-15"
        assert dob[0].score >= 0.90

    def test_negative_overrides_trigger(self, detector: RegexDetector) -> None:
        """Negative word can suppress even when trigger is present."""
        text = "The birth report filed 03/15/1990 was updated."
        entities = detector.detect(text)
        dob = [e for e in entities if e.entity_type == "DOB"]
        # "birth" (+0.40) and "filed" (-0.35) and "report" (-0.35) → score ~0.20
        assert len(dob) == 0


# ---------------------------------------------------------------------------
# Generic Date/Time detection
# ---------------------------------------------------------------------------


class TestDateAndTime:
    """Tests for generic date and timestamp extraction."""

    def test_birth_day_old_year_detected_as_date(self, detector: RegexDetector) -> None:
        """Historical birth-style dates still surface as DATE."""
        text = "Birth Day: 09/09/1752"
        entities = detector.detect(text)
        dates = [e for e in entities if e.entity_type == "DATE"]
        dobs = [e for e in entities if e.entity_type == "DOB"]
        assert len(dates) == 1
        assert dates[0].text == "09/09/1752"
        assert len(dobs) == 0

    def test_generic_date_detected(self, detector: RegexDetector) -> None:
        """Neutral calendar dates are emitted as DATE."""
        text = "Invoice date: 01/15/2026"
        entities = detector.detect(text)
        dates = [e for e in entities if e.entity_type == "DATE"]
        assert len(dates) == 1
        assert dates[0].text == "01/15/2026"
        assert dates[0].score == detector.date_score

    def test_year_first_slash_date_detected(self, detector: RegexDetector) -> None:
        """Slash-separated year-first dates are emitted as DATE."""
        text = "Actually its on 2025/05/23 for sure"
        entities = detector.detect(text)
        dates = [e for e in entities if e.entity_type == "DATE"]
        assert len(dates) == 1
        assert dates[0].text == "2025/05/23"
        assert dates[0].score == detector.date_score

    def test_dob_not_duplicated_as_date(self, detector: RegexDetector) -> None:
        """A birth-context date stays DOB and is not duplicated as DATE."""
        text = "DOB: 03/15/1990"
        entities = detector.detect(text)
        dates = [e for e in entities if e.entity_type == "DATE"]
        dobs = [e for e in entities if e.entity_type == "DOB"]
        assert len(dates) == 0
        assert len(dobs) == 1

    def test_timestamp_detected(self, detector: RegexDetector) -> None:
        """Fractional-second timestamps are emitted as TIME."""
        text = "Timestamp: 18:23:45.123"
        entities = detector.detect(text)
        times = [e for e in entities if e.entity_type == "TIME"]
        assert len(times) == 1
        assert times[0].text == "18:23:45.123"
        assert times[0].score == detector.time_score

    def test_simple_time_detected(self, detector: RegexDetector) -> None:
        """Short HH:MM times are also emitted as TIME."""
        text = "Review at 08:30 tomorrow."
        entities = detector.detect(text)
        times = [e for e in entities if e.entity_type == "TIME"]
        assert len(times) == 1
        assert times[0].text == "08:30"

    def test_textual_date_detected(self, detector: RegexDetector) -> None:
        """Textual month dates are emitted as DATE."""
        text = "The hearing is set for March 7th, 2026."
        entities = detector.detect(text)
        dates = [e for e in entities if e.entity_type == "DATE"]
        assert len(dates) == 1
        assert dates[0].text == "March 7th, 2026"

    def test_day_first_textual_date_detected(self, detector: RegexDetector) -> None:
        """Day-first textual dates are emitted as DATE."""
        text = "Follow-up on 7 March 2026 at the clinic."
        entities = detector.detect(text)
        dates = [e for e in entities if e.entity_type == "DATE"]
        assert len(dates) == 1
        assert dates[0].text == "7 March 2026"

    def test_day_first_numeric_date_detected(self, detector: RegexDetector) -> None:
        """Day-first numeric dates are emitted as DATE."""
        text = "Rescheduled for 23/05/2025 due to weather."
        entities = detector.detect(text)
        dates = [e for e in entities if e.entity_type == "DATE"]
        assert len(dates) == 1
        assert dates[0].text == "23/05/2025"

    def test_iso_datetime_yields_date_and_time(self, detector: RegexDetector) -> None:
        """ISO datetimes produce separate DATE and TIME spans."""
        text = "Logged at 2025-05-23T18:23:45.123Z by the system."
        entities = detector.detect(text)
        dates = [e for e in entities if e.entity_type == "DATE"]
        times = [e for e in entities if e.entity_type == "TIME"]
        assert len(dates) == 1
        assert dates[0].text == "2025-05-23"
        assert len(times) == 1
        assert times[0].text == "18:23:45.123Z"

    def test_twelve_hour_time_detected(self, detector: RegexDetector) -> None:
        """12-hour clock times with meridiem are emitted as TIME."""
        text = "The call starts at 6:45 PM sharp."
        entities = detector.detect(text)
        times = [e for e in entities if e.entity_type == "TIME"]
        assert len(times) == 1
        assert times[0].text == "6:45 PM"

    def test_midnight_detected(self, detector: RegexDetector) -> None:
        """Named time keywords are emitted as TIME."""
        text = "The batch rolled over at midnight."
        entities = detector.detect(text)
        times = [e for e in entities if e.entity_type == "TIME"]
        assert len(times) == 1
        assert times[0].text.lower() == "midnight"


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

    def test_undashed_format_with_trigger(self, detector: RegexDetector) -> None:
        """Plain 16-digit credit card with trigger context is detected."""
        text = "Card: 1234567890123456"
        entities = detector.detect(text)
        cc = [e for e in entities if e.entity_type == "CREDIT_CARD"]
        assert len(cc) == 1
        assert cc[0].text == "1234567890123456"
        assert cc[0].score >= 0.90

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


class TestCreditCardContext:
    """Tests for credit card undashed context-aware scoring."""

    def test_undashed_with_visa_trigger(self, detector: RegexDetector) -> None:
        """'Visa' trigger word causes undashed CC detection."""
        text = "Visa 1234567890123456 on file."
        entities = detector.detect(text)
        cc = [e for e in entities if e.entity_type == "CREDIT_CARD"]
        assert len(cc) == 1
        assert cc[0].score >= 0.90

    def test_undashed_with_credit_trigger(self, detector: RegexDetector) -> None:
        """'credit' trigger word causes undashed CC detection."""
        text = "Credit card number 1234567890123456 is saved."
        entities = detector.detect(text)
        cc = [e for e in entities if e.entity_type == "CREDIT_CARD"]
        assert len(cc) == 1
        assert cc[0].score >= 0.90

    def test_undashed_tracking_context_rejected(self, detector: RegexDetector) -> None:
        """'Tracking' context suppresses undashed CC detection."""
        text = "Tracking ID 1234567890123456 shipped."
        entities = detector.detect(text)
        cc = [e for e in entities if e.entity_type == "CREDIT_CARD"]
        assert len(cc) == 0

    def test_undashed_neutral_context_rejected(self, detector: RegexDetector) -> None:
        """Bare undashed 16-digit number with neutral context is suppressed."""
        text = "The value 1234567890123456 was recorded."
        entities = detector.detect(text)
        cc = [e for e in entities if e.entity_type == "CREDIT_CARD"]
        assert len(cc) == 0

    def test_dashed_bypasses_context(self, detector: RegexDetector) -> None:
        """Dashed CC format always gets full score regardless of context."""
        text = "Tracking 1234-5678-9012-3456 recorded."
        entities = detector.detect(text)
        cc = [e for e in entities if e.entity_type == "CREDIT_CARD"]
        assert len(cc) == 1
        assert cc[0].score == 0.99

    def test_spaced_bypasses_context(self, detector: RegexDetector) -> None:
        """Spaced CC format always gets full score regardless of context."""
        text = "Order 1234 5678 9012 3456 processed."
        entities = detector.detect(text)
        cc = [e for e in entities if e.entity_type == "CREDIT_CARD"]
        assert len(cc) == 1
        assert cc[0].score == 0.99


# ---------------------------------------------------------------------------
# Phone detection
# ---------------------------------------------------------------------------


class TestPhone:
    """Tests for domestic and international phone detection."""

    def test_us_parenthesized_phone(self, detector: RegexDetector) -> None:
        """Classic US phone format is still detected."""
        text = "Call me at (555) 123-4567 tomorrow."
        entities = detector.detect(text)
        phones = [e for e in entities if e.entity_type == "PHONE"]
        assert len(phones) == 1
        assert phones[0].text == "(555) 123-4567"

    def test_international_plus_phone_detected(self, detector: RegexDetector) -> None:
        """International +CC phone numbers are detected."""
        text = "Please feel free to contact me at +995 344 782 69 or via email at test@example.com"
        entities = detector.detect(text)
        phones = [e for e in entities if e.entity_type == "PHONE"]
        assert len(phones) == 1
        assert phones[0].text == "+995 344 782 69"

    def test_international_00_prefix_phone_detected(self, detector: RegexDetector) -> None:
        """International 00CC phone numbers are detected."""
        text = "Emergency callback: 00420 222 333 444"
        entities = detector.detect(text)
        phones = [e for e in entities if e.entity_type == "PHONE"]
        assert len(phones) == 1
        assert phones[0].text == "00420 222 333 444"

    def test_short_prefixed_number_rejected(self, detector: RegexDetector) -> None:
        """Too-short international-looking numbers are rejected."""
        text = "Ignore +44 12"
        entities = detector.detect(text)
        phones = [e for e in entities if e.entity_type == "PHONE"]
        assert len(phones) == 0

    def test_us_dashed_phone_detected(self, detector: RegexDetector) -> None:
        """Regular US dashed numbers are detected."""
        text = "Call 415-555-2671 after 5pm."
        entities = detector.detect(text)
        phones = [e for e in entities if e.entity_type == "PHONE"]
        assert len(phones) == 1
        assert phones[0].text == "415-555-2671"

    def test_us_spaced_phone_detected(self, detector: RegexDetector) -> None:
        """Regular US spaced numbers are detected."""
        text = "Alternate number 415 555 2671 is on file."
        entities = detector.detect(text)
        phones = [e for e in entities if e.entity_type == "PHONE"]
        assert len(phones) == 1
        assert phones[0].text == "415 555 2671"

    def test_local_grouped_phone_detected(self, detector: RegexDetector) -> None:
        """Grouped local numbers with trunk prefix are detected."""
        text = "Callback number 0899 862 573 was provided."
        entities = detector.detect(text)
        phones = [e for e in entities if e.entity_type == "PHONE"]
        assert len(phones) == 1
        assert phones[0].text == "0899 862 573"


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
