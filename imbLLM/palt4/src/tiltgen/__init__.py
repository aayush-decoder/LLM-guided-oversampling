"""tiltgen -- palt4: TILT, Task-Informed LLM Tilting for imbalanced tabular oversampling.

Package map (docs/MODULES.md has the long version):

    tiltgen.core          shared engine, vendored from palt3 (docs/PROVENANCE.md): data
                          protocol + column kinds, LLM switch (Qwen3 Base ladder), baselines,
                          evaluation (4 learners, quality, membership attack, stats)
    tiltgen.methods
        .tilt            THE NEW METHOD: utility models, the KL-budgeted exponential tilt,
                         SIR and field-wise twisted SMC samplers, validity + privacy gates
        .v2_corrected    the corrected ImbLLM generator TILT samples from (also a baseline)
        .v1_paper        ImbLLM as published (baseline)
        .llm_baselines   GReaT- / TapTap- / Pred-LLM-style baselines
        .pipeline        the one generator skeleton every LLM method plugs into
    tiltgen.experiments   runner, named grids (thesis_main, utility, budget, sampler, ...),
                          report (gain_vs_threshold), the no-LLM oracle study
    tiltgen.audits        theory checks (docs/THEORY.md), math, paper tables, data,
                          contamination probes
    tiltgen.cli           the single `tilt` command
"""

__version__ = "4.0.0"
