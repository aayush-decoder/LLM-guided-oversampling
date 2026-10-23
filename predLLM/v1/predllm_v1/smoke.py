"""v1 smoke test: iris, 1 seed, short fine-tune; Pred-LLM + GReaT + TapTap + Original (+ CTGAN if SDV).

v1 decodes free text, so the LLM must learn the "X is v" format before it can emit valid rows.
The default is therefore distilgpt2 with ~150 steps (seconds on GPU, a few minutes on CPU);
tiny-gpt2 is too small to learn the format (v2 has no such limit, since its decoder is schema-constrained).
"""
from predLLM.common.tabgen.llm.registry import resolve_llm
from predLLM.common.tabgen.smoke import run_smoke as _run
from predLLM.common.tabgen.utils import apply_overrides, load_yaml

from .cli import CONFIG
from .generator import VERSION, make_generator


def run_smoke(llm: str = "distilgpt2", device: str = "auto", max_steps: int = 150) -> int:
    llm = "distilgpt2" if llm == "tiny-gpt2" else llm
    cfg = apply_overrides(load_yaml(CONFIG), [f"llm={llm}", f"device={device}", f"finetune.max_steps={max_steps}",
                                              "finetune.reuse=false", "eval.runs=1", "baselines.epochs=5",
                                              "finetune.log_every=10"])
    tag = resolve_llm(llm, cfg.get("mode")).tag
    return _run(VERSION, cfg, make_generator, ["original", "pred_llm", "great", "taptap", "ctgan"], tag)
