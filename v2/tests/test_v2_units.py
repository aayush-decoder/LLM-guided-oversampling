"""v2 unit tests. The FSM / codec / curation tests are offline; the decoding tests use tiny-gpt2."""
import numpy as np
import pytest
import torch

from predllm_v2.codec import FieldCodec, Schema
from predllm_v2.curation import privacy_gate, sir_weights
from predllm_v2.decoding import NumericConstraint

SPEC = (2, 2, True)   # max 2 integer digits, 2 decimals, negatives allowed


def run(s, spec=SPEC):
    return NumericConstraint.run(NumericConstraint.START, s, spec)


@pytest.mark.parametrize("s,ok", [(" 39", True), (" -3.25", True), (" 0.5", True), (" 0.", False),
                                  (" 123", None), (" 05", None), (" 3.141", None), ("  7", True), (" -", False)])
def test_numeric_fsm(s, ok):
    st = run(s)
    if ok is None:
        assert st is None                          # rejected mid-way
    else:
        assert st is not None and NumericConstraint.accepting(st) == ok


class CharTok:
    eos_token_id, pad_token_id, name_or_path = 1, 0, "char"

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}


def test_codec_weights_value_only():
    sch = Schema(["age", "job"], "y", "classification", {"age": "num", "job": "cat", "y": "cat"},
                 {"age": 0}, {"age": (18, 90)}, {"job": ["admin", "cto"], "y": ["no", "yes"]})
    c = FieldCodec(sch, CharTok())
    ids, w = c.encode({"age": "39", "job": "cto", "y": "yes"}, ["job", "age", "y"], lam=3.0, w_scaffold=0.0)
    text = "".join(chr(i) if i > 1 else "|" for i in ids)
    assert text == "job is cto, age is 39, y is yes|"
    weighted = "".join(ch for ch, wi in zip(text, w) if wi > 0)
    assert weighted == " cto, 39, yes|"                           # only values + terminators
    assert w[-1] == 3.0 and w[text.index(" 39")] == 1.0           # lambda on the target
    assert c.prefix({"age": "39"}, ["age"], then_key="y") == [ord(ch) for ch in "age is 39, y is"]


def test_privacy_gate_and_sir():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(400, 3))
    pool = np.vstack([real[:50] + 1e-6, rng.normal(0.5, 1.0, size=(350, 3))])   # 50 near-copies
    keep, st = privacy_gate(real, pool, alpha=0.01, max_drop=0.5, seed=0)
    assert keep[:50].mean() < 0.2 and keep[50:].mean() > 0.9
    w, st = sir_weights(real, pool[50:], beta=1.0, clip=20, folds=3, seed=0)
    shifted = pool[50:, 0] > 1.5                                   # over-represented region -> lower weight
    assert w[shifted].mean() < w[~shifted].mean()


@pytest.mark.smoke
def test_cached_scoring_matches_reference_and_numbers_are_valid():
    from tabgen.llm.loader import load_tokenizer
    from transformers import AutoModelForCausalLM
    from predllm_v2.decoding import _score_no_cache, decode_numeric, score_continuations
    tok = load_tokenizer("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").eval()
    prefixes = [tok("age is 3, job is").input_ids, tok("x is").input_ids]
    cands = [tok(" admin,").input_ids, tok(" chief technology officer,").input_ids, tok(" a,").input_ids]
    a = score_continuations(model, prefixes, cands, tok.pad_token_id)
    b = _score_no_cache(model, prefixes, cands, tok.pad_token_id, 3e7)
    assert torch.allclose(a, b, atol=1e-3)
    cons = NumericConstraint(tok, model.config.vocab_size, model.device)
    sep = tok(",").input_ids[0]
    out = decode_numeric(model, [tok("age is").input_ids] * 16, SPEC, sep, cons, tok.pad_token_id, 1.0,
                         torch.Generator().manual_seed(0))
    for s in out:   # random weights, yet every finished value must parse and respect the spec
        if s is not None:
            v = float(s)
            assert abs(v) < 100 and round(v, 2) == v
