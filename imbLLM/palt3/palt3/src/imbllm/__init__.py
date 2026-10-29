"""imbllm -- ImbLLM v3 monorepo.

Package map (see docs/MODULES.md for the long version):

    imbllm.core          shared engine: data, serialisation, LLM switch, baselines,
                         evaluation, run bookkeeping, figures. NO method logic.
    imbllm.methods       the oversamplers that use an LLM
        .pipeline        the one generator skeleton both versions plug into
        .v1_paper        ImbLLM exactly as published (arXiv 2510.09783v2)
        .v2_corrected    every audit correction, each behind a named flag
        .llm_baselines   GReaT-, TapTap- and Pred-LLM-style baselines
    imbllm.experiments   the shared runner, named experiment grids, reports
    imbllm.audits        math / paper-table / data checks that call shipped code
    imbllm.cli           the single `imb` command
"""

__version__ = "3.0.0"
