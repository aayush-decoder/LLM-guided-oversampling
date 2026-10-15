# palt4: TILT, Task-Informed LLM Tilting

A B.Tech thesis codebase on **imbalanced tabular classification with LLM oversampling**.

LLM oversamplers (GReaT, Pred-LLM, ImbLLM) try to copy the minority class. palt4 starts from a limit and a question.

- **The limit** (THEORY.md, Prop. 1): at population level, even a *perfect* minority generator only moves the decision threshold. Nothing beats a tuned threshold.
- **The question:** in finite samples, *which* synthetic rows actually help?

The answer we derive is the method.

\[ q^\star(x) \;\propto\; p_{\text{LLM}}(x)\,\exp\!\big(\lambda\,u(x)\big)\,\mathbb 1[\text{valid}] ,\qquad \mathrm{KL}(q^\star\|p_{\text{LLM}})\le\varepsilon \]

- **u** is a Fisher-leverage utility: boundary × data-starved regions (Prop. 2).
- **λ** is set by a fidelity budget ε and capped for privacy (Prop. 4).
- **Sampling** is either whole-row importance resampling or **field-by-field twisted SMC**, exact for any twist (Prop. 3).
- **A validity floor** answers the label-noise objection.
- **The headline metric** is the F1 gain *over a tuned threshold*.

| read | for |
|---|---|
| [SETUP.md](SETUP.md) | install on the RTX 4050 laptop and the H100, verify, troubleshoot |
| [docs/THESIS.md](docs/THESIS.md) | the plan: RQs, contributions, chapters, timeline, risks |
| [docs/THEORY.md](docs/THEORY.md) | Props. 1–4, with status (KNOWN / NEW / CHECKED) |
| [docs/METHOD.md](docs/METHOD.md) | TILT: target, utilities, SIR / SMC, knobs, logs |
| [docs/RELATED_WORK.md](docs/RELATED_WORK.md) | the novelty claim, the closest works, reviewer objections and answers |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | E0–E9, hypotheses H1–H8, falsification conditions |
| [docs/COMPUTE.md](docs/COMPUTE.md) | RTX 4050 vs H100, the Qwen ladder, a 20 h/week plan |
| [docs/DATASETS.md](docs/DATASETS.md) | the naturally imbalanced benchmark |
| [docs/MODULES.md](docs/MODULES.md) | every module |
| [docs/PROVENANCE.md](docs/PROVENANCE.md) | what came from palt3 |
| [COMMANDS.md](COMMANDS.md) | every command, flag, experiment, metric and Make target |

## Quick start

Full installation (CUDA wheel, QLoRA, model cache, H100 sessions) is in [SETUP.md](SETUP.md). The short version:

```bash
pip install -e ".[dev]"
```

```bash
pytest -q -m "not llm and not slow"
```

```bash
tilt audit theory
```

```bash
tilt oracle --n-min 20 50 100 --seeds 30
```

The last two need no LLM and no data, and run on CPU. The oracle study is the go/no-go experiment (EXPERIMENTS.md, E0).

The LLM path:

```bash
pip install -e ".[llm,quant,dev]"
```

```bash
tilt smoke
```

```bash
tilt run thesis_main --config laptop_dev
```

```bash
tilt report thesis_main --metric gain_vs_threshold
```

## Layout

```
palt4/
├── pyproject.toml  Makefile  README.md  SETUP.md  COMMANDS.md
├── configs/
│   ├── hardware/      cpu, rtx4050_6gb, t4_16gb, p100_16gb, a100_40gb, h100_80gb
│   └── experiments/   laptop_dev, laptop_main, h100_main, h100_scale, replication, paper, smoke
├── docs/              THESIS, THEORY, METHOD, RELATED_WORK, EXPERIMENTS, COMPUTE, DATASETS,
│                      MODULES, PROVENANCE
├── src/tiltgen/
│   ├── methods/tilt/  utility · tilt · privacy · partial · smc · method      <- the new method
│   ├── methods/       pipeline · v2_corrected · v1_paper · llm_baselines      (from palt3)
│   ├── core/          data · llm · baselines · evaluate (+ privacy) · runs · serialize · viz
│   ├── experiments/   runner · grids · report · oracle
│   ├── audits/        theory_checks · math_checks · paper_tables · data_checks · contamination
│   └── cli.py         `tilt`
└── tests/             fast (no torch) + llm (offline tiny backbone)
```

## Where the LLM is used

`core/llm/engine.py` is the only module that imports transformers. `tilt where-llm --grep` lists the call sites:
- **fine-tune:** v2's corpus, shared by every TILT arm;
- **complete:** the SIR pool, and one field per particle in SMC;
- **score:** the `contrast` utility, SIR-Gibbs, and order sensitivity.

## Status (28 Sep 2026)

- The code is built.
- 104 fast tests and 6/6 theory checks pass.
- One oracle trial ran (3 seeds; no conclusion yet).
- **No LLM training and no real-table results yet.** Start with EXPERIMENTS.md, E0.
