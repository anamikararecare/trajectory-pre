"""Verbose Q1 generation with soft behavioral diagnostics.

Stance, mixed stance, partner-continuation, and target length are measured and
stored but never reject a response. Only mechanical completeness is gated.
"""

from __future__ import annotations

from src.q1 import q1_generate
from src.q1 import q1_verbose_cli  # noqa: F401 - installs progress wrappers


_measured_basic_quality = q1_generate.basic_quality
_measured_judge_quality = q1_generate.judge_quality


def _soft_basic_quality(*args, **kwargs):
    measured = _measured_basic_quality(*args, **kwargs)
    measured["within_word_target_observed"] = measured[
        "within_word_tolerance"
    ]
    # Length remains available as a covariate/quality diagnostic, but concise
    # responses do not invalidate an otherwise complete turn.
    measured["within_word_tolerance"] = True
    return measured


def _soft_judge_quality(*args, **kwargs):
    measured = _measured_judge_quality(*args, **kwargs)
    measured["diagnostic_role_consistent"] = measured["role_consistent"]
    measured["diagnostic_mixed_global_stance"] = measured[
        "mixed_global_stance"
    ]
    measured["diagnostic_continues_partner"] = measured["continues_partner"]
    # Preserve self_contained as a hard integrity gate. The other judgments
    # remain in the transcript but are deliberately non-blocking.
    measured["role_consistent"] = True
    measured["mixed_global_stance"] = False
    measured["continues_partner"] = False
    return measured


def _nonblocking_stance(role: str, score: float) -> bool:
    return True


q1_generate.basic_quality = _soft_basic_quality
q1_generate.judge_quality = _soft_judge_quality
q1_generate._role_consistent = _nonblocking_stance


if __name__ == "__main__":
    from src.q1.q1_cli import main

    main()
