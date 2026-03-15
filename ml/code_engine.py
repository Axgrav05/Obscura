from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from transformers import AutoModelForTokenClassification, pipeline

from ml.pii_engine import merge_entities
from ml.regex_detector import RegexDetector
from ml.schemas import DetectedEntity, RedactionResult

MODEL_ID = "microsoft/codebert-base"

CODE_LABELS: list[str] = [
    "O",
    "B-VAR",
    "I-VAR",
    "B-FUNC",
    "I-FUNC",
    "B-CLASS",
    "I-CLASS",
    "B-SECRET",
    "I-SECRET",
]
CODE_LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(CODE_LABELS)}
CODE_ID2LABEL: dict[int, str] = {i: label for i, label in enumerate(CODE_LABELS)}

CODE_LABEL_TO_ENTITY: dict[str, str] = {
    "VAR": "CODE_VAR",
    "FUNC": "CODE_FUNC",
    "CLASS": "CODE_CLASS",
    "SECRET": "CODE_SECRET",
}

CODE_ENTITY_TYPES: frozenset[str] = frozenset(CODE_LABEL_TO_ENTITY.values())


@dataclass
class CodeRedactionEngine:
    model_id: str = MODEL_ID
    confidence_threshold: float = 0.85
    enable_regex_fallback: bool = True
    chunk_size: int = 1500
    _pipeline: object | None = field(default=None, init=False, repr=False)
    regex_detector: RegexDetector = field(default_factory=RegexDetector)

    def load(self) -> None:
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_id,
            num_labels=len(CODE_LABELS),
            id2label=CODE_ID2LABEL,
            label2id=CODE_LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        self._pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=self.model_id,
            aggregation_strategy="simple",
            device=-1,
        )

    def _ensure_loaded(self) -> None:
        if self._pipeline is None:
            raise RuntimeError(
                "CodeRedactionEngine not loaded. Call .load() before inference."
            )

    def _detect_codebert(self, code: str) -> list[DetectedEntity]:
        results = self._pipeline(code)
        entities = []
        for ent in results:
            if ent["score"] < self.confidence_threshold:
                continue

            label_root = ent["entity_group"].split("-", 1)[-1]
            entity_type = CODE_LABEL_TO_ENTITY.get(label_root)

            if entity_type is None:
                continue

            entities.append(
                DetectedEntity(
                    text=ent["word"],
                    entity_type=entity_type,
                    start=int(ent["start"]),
                    end=int(ent["end"]),
                    score=float(ent["score"]),
                    token="",
                    source="codebert",
                )
            )
        return entities

    def detect(self, code: str) -> list[DetectedEntity]:
        self._ensure_loaded()
        code_entities = self._detect_codebert(code)

        if self.enable_regex_fallback:
            regex_entities = self.regex_detector.detect(code)
            return merge_entities(code_entities, regex_entities)

        code_entities.sort(key=lambda e: e.start)
        return code_entities

    def redact(
        self, code: str, disabled_entities: list[str] | None = None
    ) -> RedactionResult:
        entities = self.detect(code)

        if disabled_entities:
            disabled_set = frozenset(disabled_entities)
            entities = [e for e in entities if e.entity_type not in disabled_set]

        if not entities:
            return RedactionResult(
                masked_text=code,
                entities=[],
                mapping={},
                session_id=uuid.uuid4().hex,
            )

        entities.sort(key=lambda e: e.start, reverse=True)

        masked = code
        mapping = {}
        type_counters: dict[str, int] = {}

        for entity in entities:
            count = type_counters.get(entity.entity_type, 0) + 1
            type_counters[entity.entity_type] = count
            token = f"[{entity.entity_type}_{count}]"
            entity.token = token
            mapping[token] = entity.text
            masked = masked[: entity.start] + token + masked[entity.end :]

        entities.sort(key=lambda e: e.start)
        return RedactionResult(
            masked_text=masked,
            entities=entities,
            mapping=mapping,
            session_id=uuid.uuid4().hex,
        )

    @staticmethod
    def restore(masked_code: str, mapping: dict[str, str]) -> str:
        restored = masked_code
        for token, original in mapping.items():
            restored = restored.replace(token, original)
        return restored
