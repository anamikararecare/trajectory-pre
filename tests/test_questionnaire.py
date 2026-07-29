from types import SimpleNamespace

import pytest

from src.common.questionnaire import (
    aggregate_stance_responses,
    build_stance_items,
    parse_stance_response,
)


def _topic(**overrides):
    values = {
        "id": "example",
        "display_name": "the example",
        "pro_items": ["pro one", "pro two"],
        "con_items": ["con one", "con two"],
        "support_item": "I support it.",
        "oppose_item": "I oppose it.",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parser_handles_json_and_wrapped_json():
    assert parse_stance_response('{"score": 4, "confidence": 5}').score == 4
    wrapped = parse_stance_response('Answer: {"score": 2, "confidence": 3}')
    assert (wrapped.score, wrapped.confidence) == (2, 3)


def test_paper_battery_has_six_items_and_reverses_con_items():
    items = build_stance_items(_topic(), paper_compatible=True)
    assert len(items) == 6
    responses = [
        {**item, "score": 5.0 if item["direction"] == "pro" else 1.0, "confidence": 4.0}
        for item in items
    ]
    score, confidence = aggregate_stance_responses(responses)
    assert score == pytest.approx(5.0)
    assert confidence == pytest.approx(4.0)


def test_paper_battery_rejects_incomplete_topic():
    with pytest.raises(ValueError, match="lacks"):
        build_stance_items(_topic(con_items=None), paper_compatible=True)

