"""v2 smoke test: iris, 1 seed, short fine-tune; pred_llm_v2 + two ablations + Original.
Works even with the random tiny-gpt2, because v2's decoder cannot emit invalid rows."""
from predLLM.common.tabgen.llm.registry import resolve_llm
from predLLM.common.tabgen.smoke import run_smoke as _run
from predLLM.common.tabgen.utils import apply_overrides, load_yaml

from .cli import CONFIG
from .generator import VERSION, make_generator


def run_smoke(llm: str = "tiny-gpt2", device: str = "auto", max_steps: int = 30) -> int:
    cfg = apply_overrides(load_yaml(CONFIG), [f"llm={llm}", f"device={device}", f"finetune.max_steps={max_steps}",
                                              "finetune.reuse=false", "eval.runs=1", "finetune.log_every=5",
                                              "query.permutations=2"])
    tag = resolve_llm(llm, cfg.get("mode")).tag
    return _run(VERSION, cfg, make_generator, ["original", "pred_llm_v2", "v2_encoded", "v2_plain_loss"], tag)
