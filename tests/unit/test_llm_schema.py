from typing import Any

import pytest
from pydantic import ValidationError

from landedcost.classify.models import LLM_OUTPUT_SCHEMA, LlmCandidate, LlmClassification


def _objects(schema: dict[str, Any]) -> list[dict[str, Any]]:
    found = [schema] if schema.get("type") == "object" else []
    for value in schema.get("properties", {}).values():
        found += _objects(value)
    if "items" in schema:
        found += _objects(schema["items"])
    return found


def test_every_object_is_closed_and_fully_required() -> None:
    for obj in _objects(LLM_OUTPUT_SCHEMA):
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])


def test_schema_matches_pydantic_models() -> None:
    assert set(LLM_OUTPUT_SCHEMA["properties"]) == set(LlmClassification.model_fields)
    item = LLM_OUTPUT_SCHEMA["properties"]["candidates"]["items"]
    assert set(item["properties"]) == set(LlmCandidate.model_fields)
    assert "$ref" not in str(LLM_OUTPUT_SCHEMA)


def test_models_reject_extra_fields_and_bad_confidence() -> None:
    base = {"candidates": [], "confidence": 0.5, "abstained": True, "abstain_reason": "x"}
    LlmClassification.model_validate(base)
    with pytest.raises(ValidationError):
        LlmClassification.model_validate({**base, "extra": 1})
    with pytest.raises(ValidationError):
        LlmClassification.model_validate({**base, "confidence": 1.5})
