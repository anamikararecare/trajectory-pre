from src.q1 import q1_resilient_cli as resilient


def _raise(*args, **kwargs):
    raise RuntimeError("diagnostic unavailable")


def test_diagnostic_judge_failure_is_nonblocking(monkeypatch):
    monkeypatch.setattr(resilient, "_soft_judge", _raise)
    result = resilient._resilient_judge()
    assert result["self_contained"]
    assert result["diagnostic_unavailable"]
    assert "diagnostic unavailable" in result["error"]


def test_stance_battery_failure_is_recorded(monkeypatch):
    monkeypatch.setattr(resilient, "_measured_battery", _raise)
    responses, score, confidence = resilient._resilient_stance_battery()
    assert responses[0]["diagnostic_unavailable"]
    assert score is None
    assert confidence is None
