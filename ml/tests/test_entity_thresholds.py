"""Tests for per-entity semantic score thresholds in PIIEngine."""

from unittest.mock import MagicMock

from ml.pii_engine import PIIEngine


def _engine_with_pipeline(payload: list[dict]) -> PIIEngine:
    engine = PIIEngine(model_id="mock", enable_regex=False)
    engine._pipeline = MagicMock(return_value=payload)
    return engine


def test_misc_below_specific_threshold_is_dropped() -> None:
    """MISC uses a stricter threshold than the global semantic floor."""
    engine = _engine_with_pipeline(
        [
            {
                "word": "Government Support Team",
                "entity_group": "MISC",
                "start": 0,
                "end": 23,
                "score": 0.94,
            }
        ]
    )

    entities = engine.detect("Government Support Team")
    assert entities == []


def test_misc_above_specific_threshold_survives() -> None:
    """High-confidence MISC spans still survive."""
    engine = _engine_with_pipeline(
        [
            {
                "word": "American",
                "entity_group": "MISC",
                "start": 0,
                "end": 8,
                "score": 0.985,
            }
        ]
    )

    entities = engine.detect("American")
    assert len(entities) == 1
    assert entities[0].entity_type == "MISC"