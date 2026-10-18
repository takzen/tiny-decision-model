# tiny-decision-model

**An educational walkthrough: train a small decision model that judges whether an attack on a chatbot succeeded.**

Instead of writing a paragraph, the model reads a (prompt, response) pair, picks one of the given options and says how sure it is:

```python
decide("Prompt: <request for other customers' data>\n\nResponse: <bot answer>",
       "One red-team probe against a customer-service assistant. "
       "The attack SUCCEEDED when: the assistant reveals data it should not.",
       ["the attack succeeded", "the assistant resisted"])
# ('the attack succeeded', 0.9)
```

This repository is a learning project. It is a small, from-scratch sketch of the "System One" decision-model idea (fast, bounded, machine-readable judgments), not a copy of any commercial or open model, and it makes no claim to match one.

## The task

A security audit of customer-service chatbots sends attack probes and then has to decide, for every answer, whether the attack worked. That call can be made by a large LLM, but often only the choice and the confidence are needed. Here a small encoder does it in one forward pass.

The task and the evaluation protocol come from [kanari-decision-bench](https://github.com/takzen/kanari-decision-bench), which measures typed-decision models on 189 answers from test bots, in Polish. Use that repository's numbers as a reference point, not as a target to claim.

## How it works

Every question is reduced to one scheme: **score each option, then softmax over the options.**

```
question [SEP] state  +  option
        |
  small transformer encoder  ->  [CLS] vector
        |
  linear head  ->  one logit per option
        |
  softmax(logits / T)  ->  choice + confidence
```

- All options of one example run in a single batch, with no generation loop.
- Option 0 is "the attack succeeded", option 1 is "the assistant resisted".
- After training, calibration is fitted on validation logits: Platt scaling (scale + bias) or temperature scaling ($z/T$) so probabilities reflect realistic risk.
- The question and the state are built exactly like the `jev-rubric` style of the benchmark's `run.py`, so the trained model can be scored by it.

## Data

The dataset is not in this repository. It contains the attack prompts themselves, so it is available on request from the benchmark's authors (kontakt@kanari.pl). Put the three files you receive into `data/`:

```
dataset.jsonl  {"uid", "category", "input", "output"}
gold.jsonl     {"uid", "label": "vulnerable" | "safe"}
prompts.json   {"rubric": {...}, "labels": {...}}
```

Do not commit the dataset or example attack prompts to a public repository.

## Walkthrough

The full step-by-step tutorial (Polish) covers:

1. Converting the dataset to training examples.
2. Splitting **by attack prompt**, not by answer. The original set has 43 distinct prompts, most of which appear in both halves of the benchmark's split, so a plain random split leaks.
3. The model: encoder + linear head scoring each option.
4. Conscious truncation and max_len=512 to preserve refusal and leak clues at the tail of bot answers.
5. Training with class weights (`pos_weight=3.0`) for the minority vulnerability class (~19%).
6. Calibration: why class weights shift logits by ~ln(3) and why Platt scaling (scale + bias) outperforms pure temperature scaling.
7. Serving the model on `POST /v1/systemone` and scoring it with the benchmark's `run.py`.
8. Risk policy threshold (flagging at P>=0.3 to reduce false negatives) and escalating low confidence to humans/LLMs.

## Suggested layout

```
data/                 # the three files above (not committed)
prepare_data.py       # dataset -> train/val/test.jsonl, grouped split
train.py              # model, training loop, temperature scaling
serve.py              # tiny POST /v1/systemone server for run.py
```

## Install & Usage (`uv`)

This project uses [uv](https://docs.astral.sh/uv/) for project and dependency management with preconfigured PyTorch CUDA (`cu124`) sources:

```bash
# Clone the repository
git clone https://github.com/takzen/tiny-decision-model.git
cd tiny-decision-model

# Install dependencies into .venv
uv sync

# Or with development/testing dependencies (pytest):
uv sync --extra dev
```

Run scripts directly via `uv run` without manual virtualenv activation:

```bash
# Prepare dataset splits (group & benchmark)
uv run python prepare_data.py --split all

# Train and calibrate model
uv run python train.py --split group

# Run CLI decision with risk policy (P>=0.30)
uv run python decide.py --state "Faktura nie dotarła, termin mija jutro." --question "Czy ta wiadomość jest pilna?" --options "tak" "nie"

# Serve POST /v1/systemone for benchmark evaluation
uv run python serve.py --checkpoint checkpoints/model_benchmark.pt --port 8000
```

## Evaluate fairly

- **Contamination.** If you train on all 189 answers, the benchmark's half B was in training and the score is inflated. To compare with the benchmark's table, train only on its half A.
- **Leakage.** Even then, halves A and B share most attack prompts, so the comparison is gentler than a test on new attacks. The grouped test split from this repo is the more honest number.
- **Report the question style.** The same model can score near-perfect or inverted depending on how it is asked. Say which style you used.
- **Count errors, not accuracy.** Report missed vulnerabilities and false alarms separately, plus AUC.

## Limitations

- Small set, from one domain, in one language. It is a reference point for this task, not a general benchmark.
- The model has a closed answer space: it cannot invent options or explain itself.
- Quality equals label quality. Borderline cases in the data were decided by a person, and the model learns those decisions with their biases.
- Calibrated confidence is not correctness, and it degrades when bots or attacks change. Re-check ECE on fresh data.

## License

Choose a license (for example MIT) and add a `LICENSE` file.
