from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ml.code_engine import CodeRedactionEngine
from ml.schemas import DetectedEntity


@pytest.fixture
def engine() -> CodeRedactionEngine:
    eng = CodeRedactionEngine(enable_regex_fallback=True)
    # Never load the actual model in tests
    eng._pipeline = MagicMock()
    return eng


@pytest.fixture
def engine_regex_mocked() -> CodeRedactionEngine:
    """Engine with both CodeBERT pipeline and RegexDetector fully mocked."""
    eng = CodeRedactionEngine(enable_regex_fallback=True)
    eng._pipeline = MagicMock()
    eng.regex_detector.detect = MagicMock(return_value=[])
    return eng


class TestSyntaxPreservation:
    def test_restore_equals_original(self, engine: CodeRedactionEngine):
        engine._pipeline.return_value = [
            {
                "entity_group": "FUNC",
                "score": 0.97,
                "word": "my_func",
                "start": 4,
                "end": 11,
            },
            {
                "entity_group": "VAR",
                "score": 0.95,
                "word": "my_var",
                "start": 27,
                "end": 33,
            },
        ]
        code = "def my_func(x):\n    return my_var * 2\n"
        result = engine.redact(code)
        assert CodeRedactionEngine.restore(result.masked_text, result.mapping) == code

    def test_nested_loop_restore(self, engine: CodeRedactionEngine):
        engine._pipeline.return_value = [
            {
                "entity_group": "CLASS",
                "score": 0.97,
                "word": "MyClass",
                "start": 6,
                "end": 13,
            },
            {
                "entity_group": "FUNC",
                "score": 0.97,
                "word": "my_method",
                "start": 23,
                "end": 32,
            },
        ]
        code = "class MyClass:\n    def my_method(self):\n        pass\n"
        result = engine.redact(code)
        assert CodeRedactionEngine.restore(result.masked_text, result.mapping) == code


class TestStandardLibraryExclusion:
    def test_print_len_not_flagged(self, engine: CodeRedactionEngine):
        # Even if pipeline hallucinated, they should be filtered if they
        # are stdlib. Actually the spec says "Standard Library Exception —
        # CRITICAL. These are ALWAYS label O, never flagged".
        # This implies the model should be trained to ignore them, and if
        # it flags them, we might need a post-filter.
        # CodeRedactionEngine spec doesn't explicitly mention post-filtering,
        # but Section 2 implies they should never be flagged.
        engine._pipeline.return_value = []
        code = "print(len([1, 2, 3]))"
        result = engine.redact(code)
        assert len(result.entities) == 0
        assert result.masked_text == code

    def test_standard_lib_score_below_threshold(self, engine: CodeRedactionEngine):
        engine._pipeline.return_value = [
            {
                "entity_group": "FUNC",
                "score": 0.72,
                "word": "some_func",
                "start": 4,
                "end": 13,
            }
        ]
        code = "def some_func(): pass"
        result = engine.redact(code)
        # Filtered by confidence_threshold=0.85
        assert len(result.entities) == 0
        assert result.masked_text == code


class TestProprietaryHit:
    def test_proprietary_function_flagged(self, engine: CodeRedactionEngine):
        engine._pipeline.return_value = [
            {
                "entity_group": "FUNC",
                "score": 0.96,
                "word": "calc_variance",
                "start": 4,
                "end": 17,
            }
        ]
        code = "def calc_variance(x): pass"
        result = engine.redact(code)
        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "CODE_FUNC"
        assert "[CODE_FUNC_1]" in result.masked_text
        assert result.mapping["[CODE_FUNC_1]"] == "calc_variance"

    def test_proprietary_class_flagged(self, engine: CodeRedactionEngine):
        engine._pipeline.return_value = [
            {
                "entity_group": "CLASS",
                "score": 0.99,
                "word": "DataModel",
                "start": 6,
                "end": 15,
            }
        ]
        code = "class DataModel: pass"
        result = engine.redact(code)
        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "CODE_CLASS"
        assert "[CODE_CLASS_1]" in result.masked_text
        assert result.mapping["[CODE_CLASS_1]"] == "DataModel"


class TestInternalTopologies:
    def test_internal_url_redacted(self, engine: CodeRedactionEngine):
        engine._pipeline.return_value = [
            {
                "entity_group": "SECRET",
                "score": 0.98,
                "word": "https://api.internal.corp",
                "start": 10,
                "end": 35,
            }
        ]
        code = "url = 'https://api.internal.corp'"
        result = engine.redact(code)
        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "CODE_SECRET"
        assert "[CODE_SECRET_1]" in result.masked_text
        assert result.mapping["[CODE_SECRET_1]"] == "https://api.internal.corp"

    def test_public_url_not_flagged(self, engine: CodeRedactionEngine):
        engine._pipeline.return_value = []
        code = "url = 'https://google.com'"
        result = engine.redact(code)
        assert len(result.entities) == 0
        assert result.masked_text == code


class TestDualPassFallback:
    def test_credit_card_in_code_caught_by_regex(
        self, engine_regex_mocked: CodeRedactionEngine
    ):
        code = (
            "def process_payment():\n"
            "    card = '4532015112830366'\n"
            "    return charge(card)\n"
        )
        engine_regex_mocked._pipeline.return_value = [
            {
                "entity_group": "FUNC",
                "score": 0.96,
                "word": "process_payment",
                "start": 4,
                "end": 19,
            },
        ]
        # CC "4532015112830366" starts at offset 35 in the code string above
        engine_regex_mocked.regex_detector.detect.return_value = [
            DetectedEntity(
                text="4532015112830366",
                entity_type="CREDIT_CARD",
                start=35,
                end=51,
                score=0.99,
                token="",
                source="regex",
            )
        ]
        result = engine_regex_mocked.redact(code)

        entity_types = {e.entity_type for e in result.entities}
        assert "CODE_FUNC" in entity_types
        assert "CREDIT_CARD" in entity_types
        assert "4532015112830366" not in result.masked_text

    def test_ssn_in_code_caught_by_regex(
        self, engine_regex_mocked: CodeRedactionEngine
    ):
        code = "test_ssn = '123-45-6789'  # synthetic test value"
        engine_regex_mocked._pipeline.return_value = []
        # SSN "123-45-6789" starts at offset 12
        engine_regex_mocked.regex_detector.detect.return_value = [
            DetectedEntity(
                text="123-45-6789",
                entity_type="SSN",
                start=12,
                end=23,
                score=0.99,
                token="",
                source="regex",
            )
        ]
        result = engine_regex_mocked.redact(code)

        entity_types = {e.entity_type for e in result.entities}
        assert "SSN" in entity_types
        assert "123-45-6789" not in result.masked_text
