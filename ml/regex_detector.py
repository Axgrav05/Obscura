"""
Deterministic regex-based PII detector for structured entity types.

Handles detection of:
  - SSN: dashed (XXX-XX-XXXX) and dashless (XXXXXXXXX) formats.
    Dashless uses context-aware scoring to disambiguate from other 9-digit numbers.
  - PHONE: US format (NNN) NNN-NNNN and international E.164 (+CC ...).
  - EMAIL: standard user@domain.tld format.
  - MRN: Medical Record Number (MRN-XXXXXXX).
  - DOB: Dates in MM/DD/YYYY or YYYY-MM-DD format.
    Uses context-aware scoring to disambiguate birth dates from generic dates.
  - CREDIT_CARD: 16-digit credit card numbers (dashed or undashed).
    Undashed format uses context-aware scoring to reduce false positives.
  - IPV4: IPv4 addresses with 0-255 octet validation.
  - IPV6: IPv6 addresses — full, zero-compressed (::), and IPv4-mapped (::ffff:).
  - PASSPORT: US passport numbers (1 letter + 8 digits).

Latency budget: entire regex pass must complete in <0.5ms. All patterns
are pre-compiled at construction time; context scoring uses O(w) set
intersection where w is the context window size.

HIPAA/GDPR: No raw PII is logged. Only entity type and character offsets
are recorded.
"""

import re
from dataclasses import dataclass, field

from ml.schemas import DetectedEntity

# --- SSN Context Scoring Constants ---

# Trigger words that indicate a 9-digit number is likely an SSN.
# Matched against lowercased, punctuation-stripped tokens in the
# context window around the candidate match.
SSN_TRIGGER_WORDS: frozenset[str] = frozenset(
    {
        "ssn",
        "social",
        "security",
        "taxpayer",
        "tin",
        "tax",
        "identification",
        "ss#",
        "w-2",
        "w-9",
        "w2",
        "w9",
        "1099",
        "itin",
        "identity",
        "verification",
        "background",
    }
)

# Two-word phrases that provide even stronger SSN signal.
# Checked as adjacent bigrams in the context window.
SSN_TRIGGER_PHRASES: list[tuple[str, str]] = [
    ("social", "security"),
    ("tax", "id"),
    ("taxpayer", "identification"),
    ("taxpayer", "id"),
    ("background", "check"),
    ("identity", "verification"),
]

# Negative context words that suppress SSN classification.
# If these appear near a 9-digit number, it's likely a phone number,
# order number, serial number, or other non-SSN identifier.
SSN_NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "phone",
        "call",
        "fax",
        "tel",
        "telephone",
        "mobile",
        "cell",
        "dial",
        "ext",
        "order",
        "order#",
        "confirmation",
        "tracking",
        "serial",
        "account",
        "routing",
        "invoice",
        "zip",
        "postal",
        "code",
        "ref",
        "reference",
        "case#",
        "ticket",
        "po",
    }
)

# --- DOB Context Scoring Constants ---

# Trigger words that indicate a date is a birth date.
DOB_TRIGGER_WORDS: frozenset[str] = frozenset(
    {
        "dob",
        "birth",
        "born",
        "birthday",
        "birthdate",
        "age",
        "newborn",
        "neonatal",
    }
)

# Two-word phrase providing strong DOB signal.
DOB_TRIGGER_PHRASES: list[tuple[str, str]] = [
    ("date", "of"),
]

# Negative context words — these indicate non-DOB dates.
DOB_NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "meeting",
        "appointment",
        "scheduled",
        "deadline",
        "due",
        "expires",
        "expiration",
        "created",
        "updated",
        "filed",
        "issued",
        "effective",
        "invoice",
        "report",
    }
)

# --- Credit Card Context Scoring Constants ---

# Trigger words that indicate a 16-digit string is a credit card.
CC_TRIGGER_WORDS: frozenset[str] = frozenset(
    {
        "visa",
        "mastercard",
        "amex",
        "discover",
        "card",
        "credit",
        "debit",
        "cc",
        "pan",
        "cvv",
    }
)

# Negative context words for credit card.
CC_NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "id",
        "identifier",
        "tracking",
        "order",
        "receipt",
        "routing",
        "transaction",
    }
)


@dataclass
class RegexDetector:
    """Deterministic regex detector for structured PII patterns.

    Supports SSN (dashed and dashless with context disambiguation),
    PHONE (US parenthesized format), and EMAIL detection.

    Attributes:
        context_window: Number of words on each side of a match to
            examine for trigger/negative words.
        dashless_base_score: Base confidence for a raw 9-digit match
            before context scoring.
        dashless_threshold: Minimum score to emit a dashless match as SSN.
        dashed_score: Fixed confidence for dashed SSN matches.
        phone_score: Fixed confidence for phone matches.
        email_score: Fixed confidence for email matches.
    """

    context_window: int = 10
    dashless_base_score: float = 0.40
    dashless_threshold: float = 0.70
    dashed_score: float = 0.99
    phone_score: float = 0.95
    email_score: float = 0.99
    mrn_score: float = 0.99
    dob_base_score: float = 0.50
    dob_threshold: float = 0.70
    credit_card_score: float = 0.99
    cc_undashed_base_score: float = 0.50
    cc_undashed_threshold: float = 0.70
    ipv4_score: float = 0.95
    ipv6_score: float = 0.95
    passport_score: float = 0.99
    _ssn_dashed: re.Pattern[str] = field(init=False, repr=False)
    _ssn_dashless: re.Pattern[str] = field(init=False, repr=False)
    _phone: re.Pattern[str] = field(init=False, repr=False)
    _email: re.Pattern[str] = field(init=False, repr=False)
    _mrn: re.Pattern[str] = field(init=False, repr=False)
    _dob: re.Pattern[str] = field(init=False, repr=False)
    _credit_card: re.Pattern[str] = field(init=False, repr=False)
    _ipv4: re.Pattern[str] = field(init=False, repr=False)
    _ipv6: re.Pattern[str] = field(init=False, repr=False)
    _passport: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Compile regex patterns once at construction time."""
        # Dashed SSN: XXX-XX-XXXX with IRS-valid structure.
        # Lookaheads prevent 000/666/9XX area, 00 group, 0000 serial.
        # Word-character lookbehind/lookahead rejects alpha-adjacent
        # matches like "A123-45-6789B" (not real SSNs).
        self._ssn_dashed = re.compile(
            r"(?<!\w)"
            r"(?!000|666|9\d\d)(\d{3})"
            r"-"
            r"(?!00)(\d{2})"
            r"-"
            r"(?!0000)(\d{4})"
            r"(?!\w)"
        )
        # Dashless SSN: exactly 9 consecutive digits, not embedded in
        # a longer alphanumeric string.
        self._ssn_dashless = re.compile(r"(?<!\w)(\d{9})(?!\w)")

        # Phone: US (NNN) NNN-NNNN format plus international E.164.
        # International: +CC followed by 2-6 digit groups separated by
        # spaces, hyphens, or dots. Parentheses allowed around area code.
        # E.g.: +44 20 7123 4567, +1-555-010-0293, +49(0)30 123456.
        # Math expressions like "+1 - 45 = -44" are rejected because the
        # separator characters (=, *) are not in the allowed set.
        self._phone = re.compile(
            r"(?<!\w)"
            r"(?:"
            r"\(\d{3}\)\s?\d{3}-\d{4}"
            r"|"
            r"\+\d{1,3}(?:[\s\-.]?\(?\d{1,4}\)?){2,6}"
            r")"
            r"(?!\w)"
        )

        # Email: standard user@domain.tld format.
        # Greedy domain match backtracks to leave TLD as [a-zA-Z]{2,}.
        # Word-char lookahead rejects partial matches like
        # "user@ex.com123" while still excluding trailing sentence
        # punctuation (e.g. "user@ex.com.").
        self._email = re.compile(
            r"(?<!\w)" r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" r"(?!\w)"
        )

        # MRN: Medical Record Number in MRN-XXXXXXX format.
        # Case-insensitive prefix, 7-digit numeric suffix.
        self._mrn = re.compile(
            r"(?<!\w)" r"MRN-\d{7}" r"(?!\w)",
            re.IGNORECASE,
        )

        # DOB: Dates in MM/DD/YYYY or YYYY-MM-DD format.
        # Strict month (01-12), day (01-31), year (1900-2099).
        self._dob = re.compile(
            r"(?<!\w)"
            r"(?:"
            r"(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}"
            r"|"
            r"\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
            r")"
            r"(?!\w)"
        )

        # Credit Card: 16-digit numbers, dashed/spaced or undashed.
        # Dashed: 1234-5678-9012-3456 or 1234 5678 9012 3456.
        # Undashed: 1234567890123456.
        self._credit_card = re.compile(
            r"(?<!\w)" r"(?:\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}|\d{16})" r"(?!\w)"
        )

        # IPv4: Dotted quad with 0-255 octet validation.
        self._ipv4 = re.compile(
            r"(?<!\w)"
            r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
            r"(?!\w)"
        )

        # IPv6: Handles full expansion, zero-compression (::), and IPv4-mapped
        # (::ffff:a.b.c.d) forms. The IPv4-mapped alternation is listed first
        # to prevent the generic :: branch from consuming only the hex portion
        # of an IPv4-mapped address.
        #
        # MAC address rejection: MAC format (aa:bb:cc:dd:ee:ff) has 6 groups
        # with single colons only. Full IPv6 requires 8 groups (7 colons); all
        # compressed forms require double-colon (::), which MAC never has.
        self._ipv6 = re.compile(
            r"(?<!\w)"
            r"(?:"
            # IPv4-mapped: ::ffff:a.b.c.d (or ::a.b.c.d without ffff prefix)
            r"::(?:ffff(?::0{1,4})?:)?"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}"
            r"|"
            # Mixed h:h:h:h:h:h:a.b.c.d (6 hex groups + IPv4)
            r"(?:[0-9a-fA-F]{1,4}:){6}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}"
            r"|"
            # Full 8-group: h:h:h:h:h:h:h:h
            r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}"
            r"|"
            # Left groups + :: + optional right groups (left side non-empty)
            r"(?:[0-9a-fA-F]{1,4}:){1,6}"
            r":(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})?"
            r"|"
            # :: at start + optional right groups (covers ::1, ::db8:1, etc.)
            r"::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,6})?"
            r")"
            r"(?!\w)"
        )

        # US Passport: 1 uppercase letter followed by 8 digits.
        self._passport = re.compile(r"(?<!\w)[A-Z]\d{8}(?!\w)")

    def detect(self, text: str) -> list[DetectedEntity]:
        """Run all regex patterns against input text.

        Runs SSN (dashed + dashless with context scoring), PHONE, EMAIL,
        MRN, DOB, CREDIT_CARD, IPV4, and PASSPORT patterns. All matches
        are returned as DetectedEntity with source="regex".

        Args:
            text: Raw input text to scan.

        Returns:
            List of DetectedEntity objects with source="regex".
        """
        entities: list[DetectedEntity] = []
        dashed_spans = self._detect_ssn_dashed(text)
        entities.extend(dashed_spans)
        entities.extend(self._detect_ssn_dashless(text, dashed_spans))
        entities.extend(self._detect_phone(text))
        entities.extend(self._detect_email(text))
        entities.extend(self._detect_mrn(text))
        entities.extend(self._detect_dob(text))
        entities.extend(self._detect_credit_card(text))
        entities.extend(self._detect_ipv4(text))
        entities.extend(self._detect_ipv6(text))
        entities.extend(self._detect_passport(text))
        return entities

    def _detect_ssn_dashed(self, text: str) -> list[DetectedEntity]:
        """Detect dashed-format SSNs (XXX-XX-XXXX).

        Dashed format is near-unique to SSNs, so matches get a fixed
        high confidence score (0.99) without context analysis.
        """
        results: list[DetectedEntity] = []
        for match in self._ssn_dashed.finditer(text):
            area = int(match.group(1))
            group = int(match.group(2))
            serial = int(match.group(3))
            if not self._is_valid_ssn_parts(area, group, serial):
                continue
            results.append(
                DetectedEntity(
                    text=match.group(0),
                    entity_type="SSN",
                    start=match.start(),
                    end=match.end(),
                    score=self.dashed_score,
                    token="",
                    source="regex",
                )
            )
        return results

    def _detect_ssn_dashless(
        self,
        text: str,
        dashed_spans: list[DetectedEntity],
    ) -> list[DetectedEntity]:
        """Detect dashless 9-digit SSNs with context-aware disambiguation.

        A bare 9-digit number is ambiguous. We score each match using
        surrounding context words:

        score = base (0.40)
              + 0.35 if trigger word found (e.g., "ssn", "tax")
              + 0.20 if trigger phrase found (e.g., "social security")
              - 0.35 if negative word found (e.g., "phone", "order")

        Clamped to [0.0, 1.0]. Only matches above dashless_threshold
        (default 0.70) are emitted.

        Args:
            text: Raw input text.
            dashed_spans: Already-detected dashed SSNs, used to skip
                overlapping dashless matches.
        """
        results: list[DetectedEntity] = []
        for match in self._ssn_dashless.finditer(text):
            digits = match.group(1)
            area = int(digits[:3])
            group = int(digits[3:5])
            serial = int(digits[5:])

            if not self._is_valid_ssn_parts(area, group, serial):
                continue

            # Skip if this span overlaps with an already-detected dashed SSN.
            m_start, m_end = match.start(), match.end()
            if any(d.start <= m_start < d.end for d in dashed_spans):
                continue

            score = self._score_dashless_context(text, m_start, m_end)
            if score >= self.dashless_threshold:
                results.append(
                    DetectedEntity(
                        text=digits,
                        entity_type="SSN",
                        start=m_start,
                        end=m_end,
                        score=round(score, 4),
                        token="",
                        source="regex",
                    )
                )
        return results

    def _detect_phone(self, text: str) -> list[DetectedEntity]:
        """Detect US phone numbers in (NNN) NNN-NNNN format.

        Parenthesized format is distinctive enough to assign a fixed
        high confidence without context scoring.
        """
        results: list[DetectedEntity] = []
        for match in self._phone.finditer(text):
            results.append(
                DetectedEntity(
                    text=match.group(0),
                    entity_type="PHONE",
                    start=match.start(),
                    end=match.end(),
                    score=self.phone_score,
                    token="",
                    source="regex",
                )
            )
        return results

    def _detect_email(self, text: str) -> list[DetectedEntity]:
        """Detect email addresses in standard user@domain.tld format.

        Email format is unambiguous — matches get a fixed high confidence.
        Trailing sentence punctuation (e.g. "user@ex.com.") is excluded
        by the regex backtracking on the TLD match.
        """
        results: list[DetectedEntity] = []
        for match in self._email.finditer(text):
            results.append(
                DetectedEntity(
                    text=match.group(0),
                    entity_type="EMAIL",
                    start=match.start(),
                    end=match.end(),
                    score=self.email_score,
                    token="",
                    source="regex",
                )
            )
        return results

    def _detect_mrn(self, text: str) -> list[DetectedEntity]:
        """Detect Medical Record Numbers in MRN-XXXXXXX format.

        MRN format is highly distinctive and unambiguous.
        """
        results: list[DetectedEntity] = []
        for match in self._mrn.finditer(text):
            results.append(
                DetectedEntity(
                    text=match.group(0),
                    entity_type="MRN",
                    start=match.start(),
                    end=match.end(),
                    score=self.mrn_score,
                    token="",
                    source="regex",
                )
            )
        return results

    def _detect_dob(self, text: str) -> list[DetectedEntity]:
        """Detect dates of birth in MM/DD/YYYY or YYYY-MM-DD format.

        Uses context-aware scoring to disambiguate birth dates from
        generic dates (e.g., meeting dates, invoice dates). Only dates
        with sufficient contextual evidence are emitted.
        """
        results: list[DetectedEntity] = []
        for match in self._dob.finditer(text):
            m_start, m_end = match.start(), match.end()
            score = self._score_dob_context(text, m_start, m_end)
            if score >= self.dob_threshold:
                results.append(
                    DetectedEntity(
                        text=match.group(0),
                        entity_type="DOB",
                        start=m_start,
                        end=m_end,
                        score=round(score, 4),
                        token="",
                        source="regex",
                    )
                )
        return results

    def _detect_credit_card(self, text: str) -> list[DetectedEntity]:
        """Detect 16-digit credit card numbers, dashed/spaced or undashed.

        Dashed and spaced formats (e.g. 1234-5678-9012-3456) are
        distinctive enough to get a fixed high score. Undashed 16-digit
        strings use context-aware scoring to reduce false positives.
        """
        results: list[DetectedEntity] = []
        for match in self._credit_card.finditer(text):
            matched_text = match.group(0)
            m_start, m_end = match.start(), match.end()

            # Dashed/spaced formats are unambiguous — full score.
            is_undashed = matched_text.isdigit()
            if is_undashed:
                score = self._score_cc_context(text, m_start, m_end)
                if score < self.cc_undashed_threshold:
                    continue
            else:
                score = self.credit_card_score

            results.append(
                DetectedEntity(
                    text=matched_text,
                    entity_type="CREDIT_CARD",
                    start=m_start,
                    end=m_end,
                    score=round(score, 4),
                    token="",
                    source="regex",
                )
            )
        return results

    def _detect_ipv4(self, text: str) -> list[DetectedEntity]:
        """Detect IPv4 addresses with 0-255 octet validation.

        Each octet is validated to the 0-255 range in the regex itself,
        preventing false positives on version strings like 1.2.300.4.
        """
        results: list[DetectedEntity] = []
        for match in self._ipv4.finditer(text):
            results.append(
                DetectedEntity(
                    text=match.group(0),
                    entity_type="IPV4",
                    start=match.start(),
                    end=match.end(),
                    score=self.ipv4_score,
                    token="",
                    source="regex",
                )
            )
        return results

    def _detect_ipv6(self, text: str) -> list[DetectedEntity]:
        """Detect IPv6 addresses: full, zero-compressed, and IPv4-mapped forms.

        Handles:
          - Full 8-group: 2001:0db8:85a3:0000:0000:8a2e:0370:7334
          - Zero-compressed: 2001:db8::1, ::1
          - IPv4-mapped: ::ffff:192.0.2.128

        MAC addresses (aa:bb:cc:dd:ee:ff) are rejected because they have
        6 colon-separated groups with single colons only — the full IPv6
        form requires 8 groups (7 colons) and all compressed forms require
        a double-colon (::) that MAC addresses never contain.
        """
        results: list[DetectedEntity] = []
        for match in self._ipv6.finditer(text):
            results.append(
                DetectedEntity(
                    text=match.group(0),
                    entity_type="IPV6",
                    start=match.start(),
                    end=match.end(),
                    score=self.ipv6_score,
                    token="",
                    source="regex",
                )
            )
        return results

    def _detect_passport(self, text: str) -> list[DetectedEntity]:
        """Detect US passport numbers (1 uppercase letter + 8 digits).

        Standard 9-alphanumeric US passport format. Case-sensitive to
        reduce false positives on arbitrary alphanumeric strings.
        """
        results: list[DetectedEntity] = []
        for match in self._passport.finditer(text):
            results.append(
                DetectedEntity(
                    text=match.group(0),
                    entity_type="PASSPORT",
                    start=match.start(),
                    end=match.end(),
                    score=self.passport_score,
                    token="",
                    source="regex",
                )
            )
        return results

    def _score_dob_context(self, text: str, match_start: int, match_end: int) -> float:
        """Score a date match based on surrounding context.

        Examines words within self.context_window positions on each side.
        Only dates near birth-related trigger words cross the threshold.

        score = base (0.50)
              + 0.40 if trigger word found (e.g., "dob", "birth", "born")
              + 0.10 if trigger phrase found (e.g., "date of")
              - 0.35 if negative word found (e.g., "meeting", "scheduled")

        Clamped to [0.0, 1.0]. Only matches above dob_threshold
        (default 0.70) are emitted.
        """
        words_before = text[:match_start].lower().split()
        words_after = text[match_end:].lower().split()

        context_words = (
            words_before[-self.context_window :] + words_after[: self.context_window]
        )

        cleaned = [w.strip("()[]{}:;.,!?#\"'") for w in context_words]
        cleaned_set = set(cleaned)

        score = self.dob_base_score

        # Positive: trigger words
        if cleaned_set & DOB_TRIGGER_WORDS:
            score += 0.40

        # Positive: trigger phrases (bigram check)
        for w1, w2 in DOB_TRIGGER_PHRASES:
            for i in range(len(cleaned) - 1):
                if cleaned[i] == w1 and cleaned[i + 1] == w2:
                    score += 0.10
                    break
            else:
                continue
            break

        # Negative: anti-DOB context words
        if cleaned_set & DOB_NEGATIVE_WORDS:
            score -= 0.35

        return max(0.0, min(1.0, score))

    def _score_cc_context(self, text: str, match_start: int, match_end: int) -> float:
        """Score an undashed 16-digit match based on surrounding context.

        Only called for undashed credit card matches. Dashed/spaced
        formats bypass this and receive the full credit_card_score.

        score = base (0.50)
              + 0.45 if trigger word found (e.g., "visa", "card", "credit")
              - 0.35 if negative word found (e.g., "tracking", "order")

        Clamped to [0.0, 1.0]. Only matches above cc_undashed_threshold
        (default 0.70) are emitted.
        """
        words_before = text[:match_start].lower().split()
        words_after = text[match_end:].lower().split()

        context_words = (
            words_before[-self.context_window :] + words_after[: self.context_window]
        )

        cleaned = [w.strip("()[]{}:;.,!?#\"'") for w in context_words]
        cleaned_set = set(cleaned)

        score = self.cc_undashed_base_score

        # Positive: trigger words
        if cleaned_set & CC_TRIGGER_WORDS:
            score += 0.45

        # Negative: anti-CC context words
        if cleaned_set & CC_NEGATIVE_WORDS:
            score -= 0.35

        return max(0.0, min(1.0, score))

    def _score_dashless_context(
        self, text: str, match_start: int, match_end: int
    ) -> float:
        """Score a dashless 9-digit match based on surrounding context.

        Examines words within self.context_window positions on each side.
        Strips punctuation from context words before matching against
        trigger/negative word sets.

        The asymmetric weights reflect our redaction bias: a single positive
        trigger is enough to cross the threshold, but a single negative word
        can suppress it even when a trigger is present.

        Args:
            text: Full input text.
            match_start: Character start of the 9-digit match.
            match_end: Character end of the 9-digit match.

        Returns:
            Confidence score clamped to [0.0, 1.0].
        """
        words_before = text[:match_start].lower().split()
        words_after = text[match_end:].lower().split()

        context_words = (
            words_before[-self.context_window :] + words_after[: self.context_window]
        )

        # Strip punctuation for matching: "SSN:" -> "ssn", "(SSN)" -> "ssn"
        cleaned = [w.strip("()[]{}:;.,!?#\"'") for w in context_words]
        cleaned_set = set(cleaned)

        score = self.dashless_base_score

        # Positive: trigger words
        if cleaned_set & SSN_TRIGGER_WORDS:
            score += 0.35

        # Positive: trigger phrases (bigram check)
        for w1, w2 in SSN_TRIGGER_PHRASES:
            found = False
            for i in range(len(cleaned) - 1):
                if cleaned[i] == w1 and cleaned[i + 1] == w2:
                    score += 0.20
                    found = True
                    break
            if found:
                break

        # Negative: anti-SSN context words
        if cleaned_set & SSN_NEGATIVE_WORDS:
            score -= 0.35

        return max(0.0, min(1.0, score))

    @staticmethod
    def _is_valid_ssn_parts(area: int, group: int, serial: int) -> bool:
        """Validate SSN area/group/serial per IRS rules (SSA Pub 4557).

        Invalid SSNs:
        - Area: 000, 666, 900-999
        - Group: 00
        - Serial: 0000

        Args:
            area: First 3 digits (001-899, excluding 666).
            group: Middle 2 digits (01-99).
            serial: Last 4 digits (0001-9999).
        """
        if area == 0 or area == 666 or area >= 900:
            return False
        if group == 0:
            return False
        if serial == 0:
            return False
        return True
