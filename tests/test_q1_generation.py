from types import SimpleNamespace

from src.common.debate_prompts import DebateTopic
from src.q1.design import build_minimum_plan, load_protocol
from src.q1.prompts import q1_partner_message, q1_system_prompt
from src.q1.quality import basic_quality, parse_quality_json


def test_q1_minimum_plan_is_exactly_240_and_balanced():
    plan = build_minimum_plan(load_protocol("configs/q1_protocol.yaml"))
    assert len(plan) == 240
    assert plan["conv_id"].is_unique
    assert plan["condition"].value_counts().to_dict() == {
        "self_play": 128,
        "mixed_play": 112,
    }
    assert set(plan.groupby("topic_id").size()) == {30}
    assert set(plan.groupby(["topic_id", "role_a"]).size()) == {15}
    assert plan["group_model"].nunique() == 8


def test_q1_prompt_locks_stance_and_marks_partner_as_complete():
    topic = SimpleNamespace(
        display_name="the policy",
        support_item="I support the policy.",
        oppose_item="I oppose the policy.",
        pro_seed="It has benefits.",
        con_seed="It has costs.",
    )
    prompt = q1_system_prompt(topic, "opposer")
    assert "I oppose the policy." in prompt
    assert "Never endorse both global positions" in prompt
    wrapped = q1_partner_message("An unfinished-looking clause, because")
    assert "<partner_turn>" in wrapped
    assert "completed turn" in wrapped
    assert "Do not continue" in wrapped


def test_q1_basic_and_external_quality_parsing():
    text = (
        "This response keeps one position while directly answering the prior "
        "claim in a complete and independent sentence."
    )
    result = basic_quality(text, 22, 160, 10, 30)
    assert result["ends_with_terminal_punctuation"]
    assert result["within_word_tolerance"]
    assert not result["hit_token_cap"]
    parsed = parse_quality_json(
        '{"role_consistent": true, "mixed_global_stance": false, '
        '"continues_partner": false, "self_contained": true, "reason": "ok"}'
    )
    assert parsed["role_consistent"]
