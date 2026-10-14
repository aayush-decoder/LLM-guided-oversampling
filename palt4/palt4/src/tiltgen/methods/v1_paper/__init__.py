"""v1_paper -- ImbLLM exactly as published. Any deviation from the paper here is
a bug. Modules: condition (C1), permute (C2), interpolate (C3b), method (wiring,
V1Config, the Table III/IV ablation grid)."""

from .method import ABLATION, V1Config, build, strategies  # noqa: F401
