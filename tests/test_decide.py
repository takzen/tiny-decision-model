import sys
import types

import pytest


class FakeEngine:
    """Stands in for DecisionEngine so decide.py can be tested without loading a model."""

    split_type = None
    last_call = None

    def __init__(self, *args, **kwargs):
        pass

    def decide_detailed(self, state, question, options):
        FakeEngine.last_call = (state, question, options)
        return {
            "choice": options[0],
            "confidence": 0.9,
            "probabilities": {options[0]: 0.9, options[1]: 0.1},
            "p_vulnerable": 0.9,
        }


@pytest.fixture
def decide_module(monkeypatch):
    fake = types.ModuleType("src.inference")
    fake.DecisionEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "src.inference", fake)
    monkeypatch.delitem(sys.modules, "decide", raising=False)
    import decide

    return decide


def run(decide, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["decide.py", "--checkpoint", "does-not-exist.pt", *argv])
    decide.main()


def test_literal_backslash_n_becomes_newline(decide_module, monkeypatch):
    run(decide_module, monkeypatch, "--state", "Prompt: a\\n\\nResponse: b")
    state, _, _ = FakeEngine.last_call
    assert state == "Prompt: a\n\nResponse: b"


def test_defaults_describe_the_audit_task_not_a_mail_example(decide_module, monkeypatch):
    run(decide_module, monkeypatch)
    state, question, options = FakeEngine.last_call
    assert state.startswith("Prompt: ") and "\n\nResponse: " in state
    assert question.startswith("One red-team probe against a customer-service assistant.")
    assert options == ["the attack succeeded", "the assistant resisted"]


def test_missing_checkpoint_warns_that_the_head_is_untrained(decide_module, monkeypatch, capsys):
    run(decide_module, monkeypatch)
    assert "UNTRAINED" in capsys.readouterr().out
