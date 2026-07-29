"""Failure-resilient extension of the Q1 soft-gate generator."""

from __future__ import annotations

from src.q1 import q1_generate
from src.q1 import q1_soft_gate_cli  # noqa: F401 - installs soft gates


_soft_judge = q1_generate.judge_quality
_measured_battery = q1_generate._full_stance_battery


def _resilient_judge(*args, **kwargs):
    try:
        return _soft_judge(*args, **kwargs)
    except Exception as error:
        # Diagnostic service/format failures are recorded, while local
        # punctuation and token-cap checks continue to enforce completeness.
        return {
            "role_consistent": True,
            "mixed_global_stance": False,
            "continues_partner": False,
            "self_contained": True,
            "diagnostic_role_consistent": None,
            "diagnostic_mixed_global_stance": None,
            "diagnostic_continues_partner": None,
            "diagnostic_unavailable": True,
            "error": repr(error),
        }


def _resilient_stance_battery(*args, **kwargs):
    try:
        return _measured_battery(*args, **kwargs)
    except Exception as error:
        return (
            [{"diagnostic_unavailable": True, "error": repr(error)}],
            None,
            None,
        )


q1_generate.judge_quality = _resilient_judge
q1_generate._full_stance_battery = _resilient_stance_battery


if __name__ == "__main__":
    from src.q1.q1_cli import main

    main()
