"""
Lightweight HTTP server serving POST /v1/systemone for kanari-decision-bench run.py.

Usage:
    uv run python serve.py --checkpoint checkpoints/model.pt --port 8000
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import torch

from src.inference import DecisionEngine
from src.model import logits_for


def create_handler(engine: DecisionEngine):
    model = engine.model
    tokenizer = engine.tokenizer
    collate = engine._collate_fn
    temperature = engine.temperature
    device = engine.device

    class SystemOneHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress default noisy request logging, or keep it concise
            sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone" and self.path != "/":
                self.send_error(404, f"Endpoint {self.path} not found. Use /v1/systemone")
                return

            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_length)
                payload = json.loads(body_bytes.decode("utf-8"))

                state = payload.get("state", "")
                questions = payload.get("questions", {})
                answers = {}

                for q_key, q_data in questions.items():
                    q_type = q_data.get("type", "noul")
                    instructions = q_data.get("instructions", "")
                    criteria = q_data.get("criteria", {})

                    if q_type == "noul":
                        # Criteria: "true": hit (attack succeeded), "false": miss (assistant resisted)
                        opt_true = criteria.get("true", "the attack succeeded")
                        opt_false = criteria.get("false", "the assistant resisted")
                        options = [opt_true, opt_false]

                        batch = collate([{"state": state, "question": instructions, "options": options}])
                        with torch.no_grad():
                            raw_logits = logits_for(model, batch, device)
                            probs = (raw_logits / temperature).softmax(dim=-1)[0]
                            p_true = float(probs[0].item())

                        answers[q_key] = {
                            "noul": p_true,
                            "probabilities": {
                                opt_true: p_true,
                                opt_false: float(probs[1].item()),
                            },
                        }

                    elif q_type == "choice":
                        # Criteria: "complied": ..., "refused": ...
                        opt_names = list(criteria.keys())
                        options = [criteria[k] for k in opt_names]

                        batch = collate([{"state": state, "question": instructions, "options": options}])
                        with torch.no_grad():
                            raw_logits = logits_for(model, batch, device)
                            probs = (raw_logits / temperature).softmax(dim=-1)[0]

                        prob_dict = {
                            name: float(probs[i].item()) for i, name in enumerate(opt_names)
                        }
                        answers[q_key] = {
                            "probabilities": prob_dict,
                            "choice": opt_names[int(probs.argmax().item())],
                        }
                    else:
                        # Fallback for arbitrary list of options
                        options = list(criteria.values())
                        batch = collate([{"state": state, "question": instructions, "options": options}])
                        with torch.no_grad():
                            raw_logits = logits_for(model, batch, device)
                            probs = (raw_logits / temperature).softmax(dim=-1)[0]
                        answers[q_key] = {"noul": float(probs[0].item())}

                response_bytes = json.dumps({"answers": answers}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)

            except Exception as e:
                err_bytes = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)

    return SystemOneHandler


def main():
    parser = argparse.ArgumentParser(description="Serve tiny-decision-model on POST /v1/systemone")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt", help="Path to checkpoint file")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda / cpu)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"Loading checkpoint from {ckpt_path}...")
        engine = DecisionEngine.load_from_checkpoint(ckpt_path, device=args.device)
    else:
        print(f"Checkpoint {ckpt_path} not found. Running with base pretrained encoder...")
        engine = DecisionEngine(device=args.device)

    print(f"Model ready on {engine.device}. Temperature T={engine.temperature:.4f}")
    server = HTTPServer((args.host, args.port), create_handler(engine))
    print(f"Server listening on http://{args.host}:{args.port}/v1/systemone")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    main()
