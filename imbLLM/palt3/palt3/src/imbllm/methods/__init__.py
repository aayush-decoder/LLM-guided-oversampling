"""LLM-based oversamplers.

    pipeline        the shared generator skeleton + LLMSettings (backbone, device,
                    schedule, sampling) + the P8 anonymise/random-init switches
    v1_paper        ImbLLM as published
    v2_corrected    ImbLLM with named, individually-testable corrections
    llm_baselines   GReaT / TapTap / Pred-LLM style baselines on the same engine
"""
