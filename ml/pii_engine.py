"""
Obscura PII/PHI Redaction Engine

Hybrid BERT + regex NER engine for PII/PHI detection with Presidio-style
reversible masking. The BERT transformer detects semantic entities (PERSON,
ORGANIZATION, LOCATION) while deterministic regex patterns catch structured
PII (SSN, with PHONE/EMAIL/MRN planned). Results are merged via conflict
resolution rules (GAMEPLAN.md Section 2.3).

Detected entities are replaced with indexed tokens (e.g., [PERSON_1],
[SSN_2]) and tracked in a mapping dictionary for downstream restoration
by the Rust backend.

Designed for ONNX export and sub-30ms inference on AWS EC2 (x86_64).

HIPAA/GDPR: No raw PII is logged or persisted by this module.
"""

import uuid
from dataclasses import dataclass, field

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
    pipeline,
)

try:
    from ml.regex_detector import RegexDetector
    from ml.schemas import DetectedEntity, RedactionResult
except ModuleNotFoundError:
    from regex_detector import RegexDetector  # type: ignore[no-redef]
    from schemas import DetectedEntity, RedactionResult  # type: ignore[no-redef]

# Mapping from BERT NER labels to Presidio-style entity categories.
# dslim/bert-base-NER produces B-/I- prefixed IOB2 tags for:
#   PER (person), LOC (location), ORG (organization), MISC (miscellaneous)
NER_LABEL_TO_ENTITY: dict[str, str] = {
    "PER": "PERSON",
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
    "MISC": "MISC",
}

MODEL_ID = "dslim/bert-base-NER"

# Entity types where regex is authoritative (deterministic patterns).
REGEX_AUTHORITATIVE_TYPES: frozenset[str] = frozenset({"SSN", "PHONE", "EMAIL", "MRN"})

# Entity types where the transformer is authoritative (semantic).
BERT_AUTHORITATIVE_TYPES: frozenset[str] = frozenset(
    {"PERSON", "ORGANIZATION", "LOCATION", "MISC"}
)


# ---------------------------------------------------------------------------
# Hybrid conflict resolution (GAMEPLAN.md Section 2.3)
# ---------------------------------------------------------------------------


def _classify_overlap(a: DetectedEntity, b: DetectedEntity) -> str:
    """Classify the overlap relationship between two entity spans.

    Returns one of: "none", "exact", "nested", "partial".
    """
    if a.end <= b.start or b.end <= a.start:
        return "none"
    if a.start == b.start and a.end == b.end:
        return "exact"
    if (a.start <= b.start and a.end >= b.end) or (
        b.start <= a.start and b.end >= a.end
    ):
        return "nested"
    return "partial"


def _resolve_exact_overlap(a: DetectedEntity, b: DetectedEntity) -> DetectedEntity:
    """Rule 1: On exact overlap, regex wins for structured types."""
    if a.source == "regex" and a.entity_type in REGEX_AUTHORITATIVE_TYPES:
        return a
    if b.source == "regex" and b.entity_type in REGEX_AUTHORITATIVE_TYPES:
        return b
    # Both BERT or both non-structured: higher score wins.
    return a if a.score >= b.score else b


def _resolve_by_length(a: DetectedEntity, b: DetectedEntity) -> DetectedEntity:
    """Rules 2 & 3: On partial overlap or nesting, longer span wins."""
    return a if (a.end - a.start) >= (b.end - b.start) else b


def merge_entities(
    bert_entities: list[DetectedEntity],
    regex_entities: list[DetectedEntity],
) -> list[DetectedEntity]:
    """Merge entities from BERT and regex detectors with conflict resolution.

    Implements the four rules from GAMEPLAN.md Section 2.3:
      1. EXACT OVERLAP: regex wins for structured types, BERT for semantic.
      2. PARTIAL OVERLAP: take the longer span.
      3. NESTED SPANS: keep outer, discard inner.
      4. NO OVERLAP: keep both.

    Entities are processed by sorting all candidates by (start, -length)
    and greedily selecting non-conflicting spans.

    Args:
        bert_entities: Entities from the BERT NER pipeline.
        regex_entities: Entities from the regex detector.

    Returns:
        Merged, deduplicated list sorted by start offset.
    """
    if not bert_entities and not regex_entities:
        return []

    # Combine and sort by start ascending, then by span length descending.
    all_entities = bert_entities + regex_entities
    all_entities.sort(key=lambda e: (e.start, -(e.end - e.start)))

    merged: list[DetectedEntity] = []

    for candidate in all_entities:
        conflict_found = False
        for i, accepted in enumerate(merged):
            overlap = _classify_overlap(accepted, candidate)
            if overlap == "none":
                continue

            conflict_found = True
            if overlap == "exact":
                winner = _resolve_exact_overlap(accepted, candidate)
            else:
                # Rules 2 (partial) and 3 (nested): longer span wins.
                winner = _resolve_by_length(accepted, candidate)

            if winner is candidate:
                merged[i] = candidate
            break

        if not conflict_found:
            merged.append(candidate)

    merged.sort(key=lambda e: e.start)
    return merged


# ---------------------------------------------------------------------------
# PIIEngine
# ---------------------------------------------------------------------------


@dataclass
class PIIEngine:
    """Hybrid BERT + regex NER engine for PII/PHI detection and masking.

    Runs the BERT transformer for semantic entities (PERSON, ORG, LOC)
    and deterministic regex patterns for structured entities (SSN).
    Results are merged via conflict resolution rules (GAMEPLAN.md 2.3).

    Attributes:
        model_id: HuggingFace model identifier.
        device: Torch device string (cpu/cuda/mps).
        confidence_threshold: Minimum score to accept a BERT entity.
        aggregation_strategy: HF pipeline aggregation for sub-word tokens.
        enable_regex: Whether to run the regex layer.
        regex_detector: Regex detector instance for structured PII.
    """

    model_id: str = MODEL_ID
    device: str = "cpu"
    confidence_threshold: float = 0.85
    aggregation_strategy: str = "simple"
    enable_regex: bool = True
    regex_detector: RegexDetector = field(default_factory=RegexDetector)
    _pipeline: TokenClassificationPipeline | None = field(
        default=None, init=False, repr=False
    )

    def load(self) -> None:
        """Load the tokenizer and model, build the NER pipeline."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForTokenClassification.from_pretrained(self.model_id)

        device_index = -1  # CPU
        if self.device == "cuda" and torch.cuda.is_available():
            device_index = 0
        elif self.device == "mps" and torch.backends.mps.is_available():
            device_index = 0

        self._pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy=self.aggregation_strategy,
            device=device_index if device_index >= 0 else -1,
        )

    def _ensure_loaded(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("PIIEngine not loaded. Call .load() before inference.")

    def detect(self, text: str) -> list[DetectedEntity]:
        """Run hybrid NER: BERT transformer + regex patterns.

        First runs the BERT pipeline for semantic entity types (PERSON,
        ORGANIZATION, LOCATION, MISC). Then runs the regex detector for
        structured types (SSN). Finally merges results using the conflict
        resolution rules from GAMEPLAN.md Section 2.3.

        Args:
            text: Input text to analyze.

        Returns:
            Merged list of DetectedEntity sorted by character offset.
        """
        self._ensure_loaded()

        bert_entities = self._detect_bert(text)

        regex_entities: list[DetectedEntity] = []
        if self.enable_regex:
            regex_entities = self.regex_detector.detect(text)

        if regex_entities:
            return merge_entities(bert_entities, regex_entities)
        return bert_entities

    def _detect_bert(self, text: str) -> list[DetectedEntity]:
        """Run BERT NER pipeline and return entities above threshold.

        Separated from detect() to isolate BERT-specific logic from
        the hybrid orchestration.
        """
        raw_entities = self._pipeline(text)
        detected: list[DetectedEntity] = []

        for ent in raw_entities:
            score = float(ent["score"])
            if score < self.confidence_threshold:
                continue

            # Strip IOB2 prefix (B-PER -> PER, I-LOC -> LOC)
            raw_label = ent["entity_group"]
            entity_type = NER_LABEL_TO_ENTITY.get(raw_label, raw_label)

            detected.append(
                DetectedEntity(
                    text=ent["word"],
                    entity_type=entity_type,
                    start=int(ent["start"]),
                    end=int(ent["end"]),
                    score=score,
                    token="",
                    source="bert",
                )
            )

        return detected

    def redact(self, text: str) -> RedactionResult:
        """Detect PII entities and replace with reversible tokens.

        Returns the masked text plus a mapping dictionary for restoration.
        The mapping dict is the bridge artifact consumed by the Rust backend
        to restore original values after proxied LLM processing.
        """
        entities = self.detect(text)
        if not entities:
            return RedactionResult(
                masked_text=text,
                entities=[],
                mapping={},
                session_id=uuid.uuid4().hex,
            )

        # Sort by start position descending so replacements don't shift
        # earlier offsets.
        entities.sort(key=lambda e: e.start, reverse=True)

        # Count per entity type for indexed tokens: [PERSON_1], [PERSON_2]...
        type_counters: dict[str, int] = {}
        mapping: dict[str, str] = {}
        masked = text

        for entity in entities:
            count = type_counters.get(entity.entity_type, 0) + 1
            type_counters[entity.entity_type] = count
            token = f"[{entity.entity_type}_{count}]"
            entity.token = token

            mapping[token] = entity.text
            masked = masked[: entity.start] + token + masked[entity.end :]

        # Re-sort ascending for consistent output order
        entities.sort(key=lambda e: e.start)

        return RedactionResult(
            masked_text=masked,
            entities=entities,
            mapping=mapping,
            session_id=uuid.uuid4().hex,
        )

    @staticmethod
    def restore(masked_text: str, mapping: dict[str, str]) -> str:
        """Reverse the masking operation using the mapping dictionary.

        Used for testing; in production the Rust backend handles restoration.
        """
        result = masked_text
        for token, original in mapping.items():
            result = result.replace(token, original)
        return result
