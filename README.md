# tiny-decision-model

A small, calibrated decision model that **picks one of the given options instead of generating text**.

Give it a state, a question and a list of allowed answers. It returns one of them together with a probability:

```python
decide("Invoice never arrived, deadline is tomorrow.",
       "Is this message urgent?",
       ["yes", "no"])
# ('yes', 0.91)
```

This is a minimal, educational implementation of the "System One" decision-model idea: fast, bounded, machine-readable judgments that software can act on. It is not a copy of any commercial model, and it does not claim to match one.

## Why a decision model?

A large language model answers by writing, one token at a time, and you pay for every token. For many internal steps in an automated workflow (routing, classification, ranking, guardrails) you only need the choice and how sure the model is. A small model that scores a closed set of options is cheaper, faster, and its output always has a predictable shape.

## How it works

Every question is reduced to one scheme: **score each option, then softmax over the options**.

- Multiple choice: options are the labels (`invoice`, `complaint`, `question`).
- Ordinal scale: options are `1` to `5`.
- Yes/no: options are `yes` and `no`; the probability of `yes` comes from the softmax.

```
[question] [SEP] [state]  +  option
            |
      small transformer encoder  ->  [CLS] vector
            |
      linear head  ->  one logit per option
            |
      softmax(logits / T)  ->  choice + confidence
```

All options of one example run in a single batch, with no generation loop. After training, a single temperature `T` is fitted on a validation set (temperature scaling) so that "0.9" means the model is right about 90% of the time on similar inputs.

## Install

```bash
pip install torch transformers numpy
```

Default encoder: `distilbert-base-multilingual-cased` (works for English and Polish). Any small encoder from Hugging Face can be swapped in.

## Data format

One JSON object per line (`train.jsonl`, `val.jsonl`, `test.jsonl`):

```json
{"state": "Invoice never arrived, deadline is tomorrow.", "question": "Is this message urgent?", "options": ["yes", "no"], "label": 0}
```

`label` is the index of the correct option. Suggested split: 70% train, 15% validation (used for calibration), 15% test.

Labels can come from manual annotation, from distillation of a larger LLM, or a mix. If you distill, spot-check a random sample by hand: the small model inherits the labeler's mistakes and overconfidence.

## Train, calibrate, evaluate

1. Train for 2 to 4 epochs with cross-entropy over the options (AdamW, learning rate around 3e-5).
2. Fit the temperature `T` on the validation set.
3. Report accuracy and ECE (expected calibration error) on the **test** set, before and after calibration.

The full walkthrough with code is in the tutorial: *How to train your own small decision model* (link to the article).

## Use in a workflow

Add a confidence threshold. If the confidence is below it (say 0.7), hand the case to a larger LLM or a human:

```python
choice, conf = decide(text, "Is this message urgent?", ["yes", "no"])
if conf < 0.7:
    escalate(text)
```

## Limitations

- **Closed answer space.** The model cannot invent options or explain itself.
- **No multi-step reasoning.** It works for decisions based on knowledge and patterns, not for tasks that need multi-step calculation.
- **Quality equals label quality.** It will learn the biases in your data.
- **Data drift.** If the input distribution changes, confidence stops being reliable. Re-check ECE on fresh data.
- **Confidence is not correctness.** A calibrated model is still wrong sometimes, just in predictable proportions.

## Ideas for next steps

- Try a smaller or faster encoder and compare latency.
- Add more question types and measure accuracy per type.
- Benchmark against a large LLM on the same test set: accuracy, cost and time per decision.

## License

Choose a license (for example MIT) and add a `LICENSE` file.
