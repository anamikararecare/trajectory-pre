from src.q1 import q1_generate
from src.q1 import q1_soft_gate_cli as soft


def test_soft_gate_records_length_but_does_not_reject_it():
    measured = soft._soft_basic_quality(
        "A short but complete response.",
        response_tokens=7,
        max_response_tokens=160,
        min_words=30,
        max_words=80,
    )
    assert measured["within_word_tolerance"]
    assert not measured["within_word_target_observed"]
    assert measured["ends_with_terminal_punctuation"]


def test_stance_is_nonblocking():
    assert soft._nonblocking_stance("supporter", 1)
    assert soft._nonblocking_stance("opposer", 5)
