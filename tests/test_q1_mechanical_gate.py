from src.q1 import q1_mechanical_gate_cli as mechanical


def test_judge_self_containment_is_diagnostic_only(monkeypatch):
    monkeypatch.setattr(
        mechanical,
        "_diagnostic_judge",
        lambda: {
            "role_consistent": True,
            "mixed_global_stance": False,
            "continues_partner": False,
            "self_contained": False,
        },
    )
    result = mechanical._mechanical_judge()
    assert result["self_contained"]
    assert result["diagnostic_self_contained"] is False
