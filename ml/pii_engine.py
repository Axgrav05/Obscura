"""
Obscura PII/PHI Redaction Engine

Uses dslim/bert-base-NER for Named Entity Recognition with Presidio-style
reversible masking. Detected entities are replaced with indexed tokens
(e.g., [PERSON_1], [LOCATION_2]) and tracked in a mapping dictionary
for downstream restoration by the Rust backend.

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


@dataclass
class DetectedEntity:
    """A single detected PII/PHI entity with span and metadata."""

    text: str
    entity_type: str
    start: int
    end: int
    score: float
    token: str  # Reversible placeholder, e.g. [PERSON_1]


@dataclass
class RedactionResult:
    """Output of a redaction pass, including the masked text and mapping."""

    masked_text: str
    entities: list[DetectedEntity]
    mapping: dict[str, str]  # {token -> original_text}
    session_id: str


@dataclass
class PIIEngine:
    """
    BERT-based NER engine for PII/PHI detection and Presidio-style masking.

    Attributes:
        model_id: HuggingFace model identifier.
        device: Torch device string (cpu/cuda/mps).
        confidence_threshold: Minimum score to accept an entity.
        aggregation_strategy: HF pipeline aggregation for sub-word tokens.
    """

    model_id: str = MODEL_ID
    device: str = "cpu"
    confidence_threshold: float = 0.85
    aggregation_strategy: str = "simple"
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
        """
        Run NER inference and return detected entities above the
        confidence threshold.
        """
        self._ensure_loaded()

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
                    token="",  # Assigned during masking
                )
            )

        return detected

    def redact(self, text: str) -> RedactionResult:
        """
        Detect PII entities and replace them with reversible Presidio-style
        tokens. Returns the masked text plus a mapping dictionary for
        restoration.

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
        """
        Reverse the masking operation using the mapping dictionary.
        Used for testing; in production the Rust backend handles restoration.
        """
        result = masked_text
        for token, original in mapping.items():
            result = result.replace(token, original)
        return result
