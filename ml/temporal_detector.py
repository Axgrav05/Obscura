"""Broad temporal extraction for dates and times.

This module complements the deterministic regex detector with parser-backed
temporal extraction. It is designed to cover a much wider set of date and
time spellings than a small fixed regex set can realistically enumerate while
still emitting deterministic spans for the hybrid merge pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dateutil import parser

from ml.schemas import DetectedEntity


MONTH_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
DATE_SEP_PATTERN = r"[/.\-]"
NUMERIC_MONTH_PATTERN = r"(?:0?[1-9]|1[0-2])"
NUMERIC_DAY_PATTERN = r"(?:3[01]|[12]\d|0?[1-9])"
NUMERIC_YEAR_PATTERN = r"(?:\d{2}|\d{4})"
TIMEZONE_PATTERN = (
    r"(?:Z|UTC|GMT|[ECMP][SD]T|AKST|AKDT|HST|CET|CEST|"
    r"[+-]\d{2}:?\d{2})"
)
TIME_24_PATTERN = r"(?:[01]?\d|2[0-3]):[0-5]\d(?:\:[0-5]\d(?:\.\d{1,6})?)?"
TIME_12_PATTERN = (
    r"(?:0?[1-9]|1[0-2])(?:\:[0-5]\d(?:\:[0-5]\d(?:\.\d{1,6})?)?)?"
    r"\s?(?:AM|PM|am|pm)"
)
TIME_WORD_PATTERN = r"(?:noon|midnight)"
TIME_PATTERN = (
    rf"(?:{TIME_12_PATTERN}|{TIME_24_PATTERN}|{TIME_WORD_PATTERN})"
    rf"(?:\s?(?:{TIMEZONE_PATTERN}))?"
)


def _has_overlap(span: tuple[int, int], occupied_spans: list[tuple[int, int]]) -> bool:
    return any(span[0] < other_end and other_start < span[1] for other_start, other_end in occupied_spans)


@dataclass
class TemporalDetector:
    """Parser-backed detector for broad date/time coverage."""

    date_score: float = 0.94
    time_score: float = 0.93
    _datetime_patterns: list[tuple[re.Pattern[str], bool]] = field(init=False, repr=False)
    _date_patterns: list[tuple[re.Pattern[str], bool]] = field(init=False, repr=False)
    _time_pattern: re.Pattern[str] = field(init=False, repr=False)
    _date_fragment: re.Pattern[str] = field(init=False, repr=False)
    _time_fragment: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        numeric_month = r"(?:1[0-2]|0?[1-9])"
        numeric_mdy = (
            rf"{numeric_month}{DATE_SEP_PATTERN}{NUMERIC_DAY_PATTERN}"
            rf"{DATE_SEP_PATTERN}{NUMERIC_YEAR_PATTERN}"
        )
        numeric_dmy = (
            rf"{NUMERIC_DAY_PATTERN}{DATE_SEP_PATTERN}{numeric_month}"
            rf"{DATE_SEP_PATTERN}{NUMERIC_YEAR_PATTERN}"
        )
        numeric_ymd = (
            rf"(?:\d{{4}}){DATE_SEP_PATTERN}{numeric_month}"
            rf"{DATE_SEP_PATTERN}{NUMERIC_DAY_PATTERN}"
        )
        text_mdy = (
            rf"{MONTH_PATTERN}[ \t]+{NUMERIC_DAY_PATTERN}(?:st|nd|rd|th)?"
            rf"(?:,?[ \t]+{NUMERIC_YEAR_PATTERN})?"
        )
        text_dmy = (
            rf"{NUMERIC_DAY_PATTERN}(?:st|nd|rd|th)?[ \t]+{MONTH_PATTERN}"
            rf"(?:,?[ \t]+{NUMERIC_YEAR_PATTERN})?"
        )

        self._date_patterns = [
            (re.compile(rf"(?<!\w){numeric_ymd}(?!\w)"), False),
            (re.compile(rf"(?<!\w){numeric_mdy}(?!\w)"), False),
            (re.compile(rf"(?<!\w){numeric_dmy}(?!\w)"), True),
            (re.compile(rf"(?<!\w){text_mdy}(?!\w)"), False),
            (re.compile(rf"(?<!\w){text_dmy}(?!\w)"), True),
        ]
        self._datetime_patterns = [
            (
                re.compile(
                    rf"(?<!\w){numeric_ymd}(?:[T ,]+){TIME_PATTERN}(?!\w)"
                ),
                False,
            ),
            (
                re.compile(
                    rf"(?<!\w){numeric_mdy}(?:[T ,]+){TIME_PATTERN}(?!\w)"
                ),
                False,
            ),
            (
                re.compile(
                    rf"(?<!\w){numeric_dmy}(?:[T ,]+){TIME_PATTERN}(?!\w)"
                ),
                True,
            ),
            (
                re.compile(
                    rf"(?<!\w){text_mdy}(?:[T ,]+){TIME_PATTERN}(?!\w)"
                ),
                False,
            ),
            (
                re.compile(
                    rf"(?<!\w){text_dmy}(?:[T ,]+){TIME_PATTERN}(?!\w)"
                ),
                True,
            ),
        ]
        self._time_pattern = re.compile(rf"(?<!\w){TIME_PATTERN}(?!\w)")
        self._date_fragment = re.compile(
            rf"{numeric_ymd}|{numeric_mdy}|{numeric_dmy}|{text_mdy}|{text_dmy}",
            re.IGNORECASE,
        )
        self._time_fragment = re.compile(TIME_PATTERN, re.IGNORECASE)

    def detect(
        self,
        text: str,
        occupied_spans: list[tuple[int, int]] | None = None,
    ) -> list[DetectedEntity]:
        """Detect additional DATE and TIME entities using parser validation."""
        occupied = list(occupied_spans or [])
        results: list[DetectedEntity] = []

        for pattern, dayfirst in self._datetime_patterns:
            for match in pattern.finditer(text):
                candidate = match.group(0)
                if not self._can_parse_datetime(candidate, dayfirst=dayfirst):
                    continue

                date_match = self._date_fragment.search(candidate)
                time_match = None
                if date_match is not None:
                    time_match = self._time_fragment.search(candidate, date_match.end())
                if date_match is not None:
                    span = (
                        match.start() + date_match.start(),
                        match.start() + date_match.end(),
                    )
                    if not _has_overlap(span, occupied) and self._can_parse_date(
                        date_match.group(0), dayfirst=dayfirst
                    ):
                        results.append(
                            DetectedEntity(
                                text=text[span[0] : span[1]],
                                entity_type="DATE",
                                start=span[0],
                                end=span[1],
                                score=self.date_score,
                                token="",
                                source="regex",
                            )
                        )
                        occupied.append(span)

                if time_match is not None:
                    span = (
                        match.start() + time_match.start(),
                        match.start() + time_match.end(),
                    )
                    if not _has_overlap(span, occupied) and self._can_parse_time(
                        time_match.group(0)
                    ):
                        results.append(
                            DetectedEntity(
                                text=text[span[0] : span[1]],
                                entity_type="TIME",
                                start=span[0],
                                end=span[1],
                                score=self.time_score,
                                token="",
                                source="regex",
                            )
                        )
                        occupied.append(span)

        for pattern, dayfirst in self._date_patterns:
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                if _has_overlap(span, occupied):
                    continue
                if not self._can_parse_date(match.group(0), dayfirst=dayfirst):
                    continue
                results.append(
                    DetectedEntity(
                        text=match.group(0),
                        entity_type="DATE",
                        start=match.start(),
                        end=match.end(),
                        score=self.date_score,
                        token="",
                        source="regex",
                    )
                )
                occupied.append(span)

        for match in self._time_pattern.finditer(text):
            span = (match.start(), match.end())
            if _has_overlap(span, occupied):
                continue
            if not self._can_parse_time(match.group(0)):
                continue
            results.append(
                DetectedEntity(
                    text=match.group(0),
                    entity_type="TIME",
                    start=match.start(),
                    end=match.end(),
                    score=self.time_score,
                    token="",
                    source="regex",
                )
            )
            occupied.append(span)

        return results

    @staticmethod
    def _can_parse_date(candidate: str, *, dayfirst: bool) -> bool:
        try:
            parser.parse(candidate, fuzzy=False, dayfirst=dayfirst)
            return True
        except (ValueError, OverflowError):
            return False

    @staticmethod
    def _can_parse_time(candidate: str) -> bool:
        if candidate.lower() in {"noon", "midnight"}:
            return True
        try:
            parser.parse(candidate, fuzzy=False)
            return True
        except (ValueError, OverflowError):
            return False

    @staticmethod
    def _can_parse_datetime(candidate: str, *, dayfirst: bool) -> bool:
        try:
            parser.parse(candidate, fuzzy=False, dayfirst=dayfirst)
            return True
        except (ValueError, OverflowError):
            return False