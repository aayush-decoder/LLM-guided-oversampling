# tabgen: code shared by v1 (paper replication) and v2 (improved method).

# data/       registry, download, cleaning, paper-protocol encoding, train/test splits
# llm/        LLM switch (registry + loader for full/LoRA/QLoRA) and the shared fine-tuning loop
# eval/       TSTR (XGBoost), fidelity metrics (discriminator, inverse KL, density, coverage, DCR), tables
# baselines/  SDV CTGAN / TVAE / CopulaGAN
# viz/        data-stage plots, synthetic-vs-real plots, attention plot (Fig. 4)
# experiment  version-agnostic runner that writes <version>/runs/...
# audit       env / data / run audits
__version__ = "1.0.0"
