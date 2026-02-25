"""
Shared data types for the Obscura PII/PHI detection pipeline.

Extracted to avoid circular imports between pii_engine.py and
regex_detector.py. Both modules import these types.
"""

from dataclasses import dataclass


@dataclass
class DetectedEntity:
    """A single detected PII/PHI entity with span and metadata.

    Attributes:
        text: The original surface form of the entity.
        entity_type: Canonical type (PERSON, SSN, LOCATION, etc.).
        start: Character offset of the entity start in the original text.
        end: Character offset of the entity end in the original text.
        score: Confidence score (0.0-1.0). For BERT, this is the model's
            softmax probability. For regex, this is 0.99 (dashed SSN) or
            context-scored (dashless SSN).
        token: Reversible placeholder, e.g. [PERSON_1]. Assigned during masking.
        source: Which detector produced this entity: "bert" or "regex".
    """

    text: str
    entity_type: str
    start: int
    end: int
    score: float
    token: str  # Reversible placeholder, e.g. [PERSON_1]
    source: str = "bert"  # "bert" or "regex"


@dataclass
class RedactionResult:
    """Output of a redaction pass, including the masked text and mapping."""

    masked_text: str
    entities: list[DetectedEntity]
    mapping: dict[str, str]  # {token -> original_text}
    session_id: str
