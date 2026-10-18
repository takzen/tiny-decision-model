import random

import prepare_data as pd


def make_examples(n=189, n_groups=43, n_vuln=36, seed=1):
    """Synthetic examples shaped like the audit set: many answers share a few attack prompts."""
    rnd = random.Random(seed)
    cats = [f"cat{i}" for i in range(10)]
    examples = []
    for i in range(n):
        g = rnd.randrange(n_groups)
        examples.append(
            {
                "uid": f"u{i}",
                "category": cats[g % 10],
                "group": f"prompt-{g}",
                "label": 0 if i < n_vuln else 1,  # 0 = vulnerable
            }
        )
    rnd.shuffle(examples)
    return examples


def test_dummy_data_creates_missing_data_dir(tmp_path):
    target = tmp_path / "data"  # does not exist, like data/ after a fresh clone
    pd.create_sample_dummy_data(target)
    assert (target / "sample_dummy.jsonl").exists()


def test_group_split_has_no_prompt_leakage():
    splits = pd.split_stratified_group(make_examples())
    groups = {name: {e["group"] for e in items} for name, items in splits.items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]


def test_group_split_keeps_every_example_once():
    examples = make_examples()
    splits = pd.split_stratified_group(examples)
    uids = [e["uid"] for items in splits.values() for e in items]
    assert sorted(uids) == sorted(e["uid"] for e in examples)


def test_group_split_puts_vulnerable_answers_in_every_part():
    splits = pd.split_stratified_group(make_examples())
    for name, items in splits.items():
        assert any(e["label"] == 0 for e in items), name


def test_benchmark_split_reproduces_run_py_half_b():
    examples = make_examples()
    gold = {e["uid"]: ("vulnerable" if e["label"] == 0 else "safe") for e in examples}

    # the split logic of kanari-decision-bench run.py, written out independently
    rnd = random.Random(7)
    uids = sorted(gold)
    vuln = sorted(u for u in uids if gold[u] == "vulnerable")
    safe = sorted(u for u in uids if gold[u] == "safe")
    rnd.shuffle(vuln)
    rnd.shuffle(safe)
    half_a = set(vuln[: len(vuln) // 2] + safe[: len(safe) // 2])
    expected_b = {u for u in uids if u not in half_a}

    splits = pd.split_benchmark(examples, gold)
    assert {e["uid"] for e in splits["test"]} == expected_b
    assert not {e["uid"] for e in splits["train"] + splits["val"]} & expected_b
    assert len(splits["test"]) == 95
