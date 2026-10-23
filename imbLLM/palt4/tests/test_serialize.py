"""Serialisation, parsing, repair rules, parse accounting."""

from __future__ import annotations

from tiltgen.core.serialize import (ParseStats, category_vocab, fmt_num, prompt_from_clauses,
                                   row_to_text, text_to_row, texts_to_frame)


def test_roundtrip(splits):
    s = splits.schema
    vocab = category_vocab(splits.imbalanced_train, s)
    row = splits.train_minor.iloc[0]
    text = row_to_text(row, [s.target, *s.features], s)
    back = text_to_row(text, s, vocab=vocab)
    assert back is not None
    for c in s.categorical:
        assert back[c] == row[c]
    for c in s.continuous:
        assert abs(back[c] - float(row[c])) <= 1e-4 * max(1, abs(float(row[c])))


def test_numbers_keep_significant_digits():
    assert fmt_num(0.078712345) == "0.078712"          # palt2: "0.08"
    assert fmt_num(30162.0) == "30162"
    assert fmt_num(-1.5) == "-1.5"


def test_prompts_end_with_comma_not_space():
    p = prompt_from_clauses(["target is yes", "age is 30"])
    assert p.endswith(",") and not p.endswith(", ")


def test_repair_rejects_empty_and_ambiguous(splits):
    s = splits.schema
    vocab = {"edu": ["bachelor", "master", "school"], "city": ["north", "south"]}
    base = "target is yes, age is 40, income is 5, score is 0.5, city is north, edu is {}"
    assert text_to_row(base.format("Bachelor"), s, vocab=vocab, repair=True)["edu"] == "bachelor"
    assert text_to_row(base.format(""), s, vocab=vocab, repair=True) is None       # palt2: 'bachelor'
    assert text_to_row(base.format("s"), s, vocab=vocab, repair=True) is None      # too short
    assert text_to_row(base.format("mast"), s, vocab=vocab, repair=True)["edu"] == "master"
    assert text_to_row(base.format("Bachelor"), s, vocab=vocab, repair=False) is None


def test_yield_is_parsed_over_completions(splits):
    s = splits.schema
    good = row_to_text(splits.train_minor.iloc[0], [s.target, *s.features], s)
    st = ParseStats(prompts_sent=10, completions=10)
    frame, keep = texts_to_frame([good] * 6 + ["junk"] * 4, s,
                                 vocab=category_vocab(splits.imbalanced_train, s), stats=st)
    assert len(frame) == 6 and keep == list(range(6))
    assert st.yield_rate == 0.6 and st.dropped_missing_features == 4


def test_parser_takes_first_occurrence_and_ignores_unknown_columns(splits):
    s = splits.schema
    t = ("target is yes, age is 40, bogus is 1, income is 5, score is 0.5, city is north, "
         "edu is school, age is 99")
    st = ParseStats()
    r = text_to_row(t, s, vocab=category_vocab(splits.imbalanced_train, s), stats=st)
    assert r["age"] == 40.0 and st.unknown_column_clauses == 1
