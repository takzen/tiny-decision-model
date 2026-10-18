import json
import threading
from http.server import HTTPServer
import urllib.request
from src.inference import DecisionEngine
from tests.test_inference import DummyDecisionModel
from tests.test_model import MockTokenizer
from serve import create_handler


def test_serve_systemone():
    tokenizer = MockTokenizer()
    model = DummyDecisionModel()
    engine = DecisionEngine(
        model=model,
        tokenizer=tokenizer,
        temperature=1.0,
        model_name="dummy",
        device="cpu",
    )

    server = HTTPServer(("127.0.0.1", 0), create_handler(engine))
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    payload = {
        "state": "Prompt: test prompt\n\nResponse: test bot response",
        "questions": {
            "q": {
                "type": "noul",
                "instructions": "One red-team probe. The attack SUCCEEDED when: test rubric",
                "criteria": {
                    "true": "the attack succeeded",
                    "false": "the assistant resisted",
                },
            }
        },
    }

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/systemone",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))

    server.shutdown()

    assert "answers" in data
    assert "q" in data["answers"]
    assert "noul" in data["answers"]["q"]
    assert 0.0 <= data["answers"]["q"]["noul"] <= 1.0
