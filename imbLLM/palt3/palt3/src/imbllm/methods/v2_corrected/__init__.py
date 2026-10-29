"""v2_corrected -- ImbLLM with every audit correction behind a named flag.
Modules: interpolate (geometry + anchor categoricals), condition (tau-bridge),
gibbs (order sensitivity + SIR-Gibbs repair), method (V2Config, presets, wiring)."""

from .method import PRESETS, V1_VALUES, V2_VALUES, V2Config, build, strategies  # noqa: F401
