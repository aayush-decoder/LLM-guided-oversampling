"""LLM layer: zoo (which backbones are allowed) -> hardware (how they run on this
device) -> engine (the only code that calls transformers)."""

from .hardware import Hardware, TrainPlan, detect, get_hardware, plan  # noqa: F401
from .zoo import DEFAULT, MODELS, ModelSpec, get  # noqa: F401
