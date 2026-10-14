"""The ONLY module that talks to a language model. Three verbs:

    finetune(texts)                      causal-LM training on serialised rows
    complete(prompts)                    sample row continuations
    score(contexts, suffixes)            sum log p(suffix tokens | context)

Everything else (v1, v2, the LLM baselines) calls these; `tilt where-llm` lists
the call sites. torch / transformers / peft are imported lazily so data, audits,
classical baselines and most tests run without them.

CORRECTIONS vs palt2 (see docs/what_corrected.md, section A)
  * precision: trainable params are fp32 (palt2 loaded fp16 weights and then
    ran GradScaler over fp16 grads -> "Attempting to unscale FP16 gradients")
  * padding: RIGHT padding for training, LEFT only for generation (palt2 left-
    padded training batches, shifting GPT-2's absolute positions per row)
  * EOS is appended to every training row, so the model learns where a row
    ends and generation stops instead of running to max_new_tokens
  * scoring sums log-probs over the value tokens only (+ terminator), from
    token ids built as ctx_ids + suffix_ids so every candidate is scored
    under the same context tokenisation (palt2 averaged over the whole
    clause and re-tokenised the joined string)
"""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .hardware import TrainPlan
from .zoo import ModelSpec, check_access


@dataclass
class TrainCfg:
    epochs: int = 50               # paper
    lr: float = 5e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_len: int = 512             # hard cap; rows longer than this are counted
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 0
    log_every: int = 0


# --------------------------------------------------------------------------- tiny

def build_byte_tokenizer():
    """Byte-level tokenizer with no merges: 256 symbols + <|endoftext|>.
    Built in memory -- no download -- for the tiny-random smoke backbone."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    vocab["<|endoftext|>"] = len(vocab)
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens(["<|endoftext|>"])
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>",
                                   pad_token="<|endoftext|>")


def _tiny_model(tok):
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(vocab_size=len(tok), n_positions=2048, n_embd=128, n_layer=4,
                     n_head=4, bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    return GPT2LMHeadModel(cfg)


def _dtype_kw(dtype) -> dict:
    import transformers
    from packaging.version import Version
    key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    return {key: dtype}


# --------------------------------------------------------------------------- engine

class LLMEngine:
    def __init__(self, spec: ModelSpec, plan: TrainPlan, *, init: str = "pretrained",
                 seed: int = 0):
        if not plan.fits:
            raise RuntimeError(f"Refusing to start: {plan.reason}")
        if init not in ("pretrained", "random"):
            raise ValueError("init must be 'pretrained' or 'random'")
        check_access(spec)
        self.spec, self.plan, self.init, self.seed = spec, plan, init, seed
        self.model = None
        self.tok = None
        self.stats: dict = {"train_tokens": 0, "gen_tokens": 0, "score_tokens": 0,
                            "rows_truncated": 0}

    # ------------------------------------------------------------ plumbing
    @property
    def device(self) -> str:
        return self.plan.device

    def _torch_dtype(self):
        import torch
        return {"bf16": torch.bfloat16, "fp16": torch.float16,
                "fp32": torch.float32}[self.plan.precision]

    def _autocast(self):
        import contextlib

        import torch
        if self.device != "cuda" or self.plan.precision == "fp32":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self._torch_dtype())

    def _load_tokenizer(self, src: str | None = None):
        if self.spec.family == "tiny" and src is None:
            tok = build_byte_tokenizer()
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(src or self.spec.hf_id)
        if tok.eos_token is None:
            raise RuntimeError(f"{self.spec.key}: tokenizer has no EOS token")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _base_model(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        if self.spec.family == "tiny":
            torch.manual_seed(self.seed)
            return _tiny_model(self.tok)
        kw: dict = {"attn_implementation": self.plan.attn}
        if self.plan.mode == "full" or self.device != "cuda":
            kw.update(_dtype_kw(torch.float32))          # fp32 master weights
        elif self.plan.mode == "lora":
            kw.update(_dtype_kw(self._torch_dtype()))    # frozen half-precision base
        else:                                            # qlora
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=self._torch_dtype())
            kw["device_map"] = {"": 0}
        if self.init == "random":                        # P8: same architecture, no prior
            cfg = AutoConfig.from_pretrained(self.spec.hf_id)
            torch.manual_seed(self.seed)
            kw.pop("quantization_config", None)
            kw.pop("device_map", None)
            if self.plan.mode != "full" and self.device == "cuda":
                kw.update(_dtype_kw(self._torch_dtype()))
            return AutoModelForCausalLM.from_config(cfg, **kw)
        return AutoModelForCausalLM.from_pretrained(self.spec.hf_id, **kw)

    @property
    def _uses_device_map(self) -> bool:
        return self.plan.mode == "qlora" and self.init == "pretrained" \
            and self.spec.family != "tiny" and self.device == "cuda"

    def load(self, cache_dir: Path | None = None) -> "LLMEngine":
        """Fresh backbone, or a fine-tuned one from the model cache."""
        if self.model is not None:
            return self
        import torch
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if cache_dir is not None and (Path(cache_dir) / "engine.json").exists():
            meta = json.loads((Path(cache_dir) / "engine.json").read_text())
            self.tok = self._load_tokenizer(str(cache_dir))
            if meta["mode"] in ("lora", "qlora"):
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self._base_model(), str(cache_dir))
            else:
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(cache_dir), **_dtype_kw(torch.float32))
        else:
            self.tok = self._load_tokenizer()
            self.model = self._base_model()
            if self.plan.mode in ("lora", "qlora") and self.spec.family != "tiny":
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
                if self._uses_device_map:
                    self.model = prepare_model_for_kbit_training(
                        self.model, use_gradient_checkpointing=self.plan.grad_ckpt)
                self.model = get_peft_model(self.model, LoraConfig(
                    r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                    task_type="CAUSAL_LM", bias="none"))
        if self.plan.grad_ckpt and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        if not self._uses_device_map:                     # 4-bit models are placed by device_map
            self.model.to(self.device)
        return self

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)
        (out_dir / "engine.json").write_text(json.dumps(
            {"model": self.spec.key, "mode": self.plan.mode, "init": self.init,
             "plan": self.plan.as_dict()}, indent=2))

    def release(self) -> None:
        self.model = None
        self.tok = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------ tokens
    def encode(self, texts: list[str], *, eos: bool) -> list[list[int]]:
        ids = self.tok(texts, add_special_tokens=True)["input_ids"]
        return [x + [self.tok.eos_token_id] if eos else x for x in ids]

    def token_lengths(self, texts: list[str]) -> list[int]:
        self.load()
        return [len(x) for x in self.encode(texts, eos=True)]

    # ------------------------------------------------------------ 1. fine-tune
    def finetune(self, texts: list[str], cfg: TrainCfg) -> list[dict]:
        import torch
        self.load()
        torch.manual_seed(cfg.seed)
        ids = self.encode(texts, eos=True)
        trunc = sum(len(x) > cfg.max_len for x in ids)
        self.stats["rows_truncated"] += trunc
        ids = [x[:cfg.max_len] for x in ids]
        pad = self.tok.pad_token_id
        mb, accum = self.plan.micro_batch, self.plan.grad_accum

        params = [p for p in self.model.parameters() if p.requires_grad]
        if any(p.dtype != torch.float32 for p in params):
            raise RuntimeError("trainable parameters must be fp32 (mixed precision via "
                               "autocast only)")
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        steps_per_epoch = math.ceil(len(ids) / (mb * accum))
        total = max(1, steps_per_epoch * cfg.epochs)
        warm = max(1, int(cfg.warmup_ratio * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(
                math.pi * min(1.0, max(0, s - warm) / max(1, total - warm)))))
        scaler = torch.amp.GradScaler("cuda", enabled=(self.device == "cuda"
                                                       and self.plan.precision == "fp16"))
        g = torch.Generator().manual_seed(cfg.seed)
        history = []
        self.model.train()
        for ep in range(cfg.epochs):
            t0, run, nb, ntok = time.time(), 0.0, 0, 0
            order = torch.randperm(len(ids), generator=g).tolist()
            batches = [order[i:i + mb] for i in range(0, len(order), mb)]
            opt.zero_grad(set_to_none=True)
            for bi, b in enumerate(batches):
                L = max(len(ids[i]) for i in b)
                inp = torch.full((len(b), L), pad, dtype=torch.long)
                att = torch.zeros((len(b), L), dtype=torch.long)
                for r, i in enumerate(b):                  # RIGHT padding
                    inp[r, :len(ids[i])] = torch.tensor(ids[i])
                    att[r, :len(ids[i])] = 1
                lab = inp.masked_fill(att == 0, -100)
                inp, att, lab = inp.to(self.device), att.to(self.device), lab.to(self.device)
                with self._autocast():
                    loss = self.model(input_ids=inp, attention_mask=att, labels=lab).loss
                scaler.scale(loss / accum).backward()
                run += loss.detach().item()
                nb += 1
                ntok += int(att.sum())
                if (bi + 1) % accum == 0 or bi == len(batches) - 1:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
            self.stats["train_tokens"] += ntok
            history.append({"epoch": ep + 1, "loss": run / max(nb, 1), "tokens": ntok,
                            "sec": round(time.time() - t0, 2),
                            "lr": opt.param_groups[0]["lr"]})
            if cfg.log_every and (ep + 1) % cfg.log_every == 0:
                print(f"    epoch {ep + 1:>3}/{cfg.epochs} loss {history[-1]['loss']:.4f}",
                      flush=True)
        self.model.eval()
        return history

    # ------------------------------------------------------------ 2. sample
    def complete(self, prompts: list[str], *, temperature: float = 0.7, top_p: float = 1.0,
                 max_new_tokens: int = 256, batch_size: int = 32,
                 seed: int = 0, greedy: bool = False) -> list[tuple[str, bool]]:
        """-> [(continuation text, ended_with_eos)] in prompt order.
        `greedy=True` decodes deterministically (contamination probes)."""
        import torch
        self.load()
        self.model.eval()
        torch.manual_seed(seed)
        eos = self.tok.eos_token_id
        out: list[tuple[str, bool]] = [("", False)] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        self.tok.padding_side = "left"                     # LEFT only for generation
        try:
            for s in range(0, len(order), batch_size):
                idx = order[s:s + batch_size]
                enc = self.tok([prompts[i] for i in idx], return_tensors="pt",
                               padding=True).to(self.device)
                with torch.inference_mode(), self._autocast():
                    samp = ({"do_sample": False} if greedy else
                            {"do_sample": True, "temperature": temperature, "top_p": top_p,
                             "top_k": 0})
                    gen = self.model.generate(
                        **enc, **samp, max_new_tokens=max_new_tokens,
                        pad_token_id=self.tok.pad_token_id, eos_token_id=eos)
                new = gen[:, enc["input_ids"].shape[1]:]
                self.stats["gen_tokens"] += int((new != self.tok.pad_token_id).sum())
                texts = self.tok.batch_decode(new, skip_special_tokens=True)
                for k, i in enumerate(idx):
                    out[i] = (texts[k], bool((new[k] == eos).any()))
        finally:
            self.tok.padding_side = "right"
        return out

    # ------------------------------------------------------------ 3. score
    def score(self, contexts: list[str], suffixes: list[list[str]], *, add_eos: bool = False,
              batch_size: int = 64) -> list[list[float]]:
        """sum_t log p(suffix_t | context, suffix_<t)  for every (context, suffix).

        Token ids are ctx_ids + suffix_ids (+ eos). A suffix should start with a
        space (" 40.5") so that, under byte-level BPE, the split point is a
        natural token boundary and matches training tokenisation."""
        import torch
        self.load()
        self.model.eval()
        ctx_ids = self.tok(contexts, add_special_tokens=True)["input_ids"]
        seqs, spans, owner = [], [], []
        for ci, (c, sufs) in enumerate(zip(ctx_ids, suffixes)):
            if not sufs:
                continue
            s_ids = self.tok(list(sufs), add_special_tokens=False)["input_ids"]
            for si, s in enumerate(s_ids):
                full = c + s + ([self.tok.eos_token_id] if add_eos else [])
                seqs.append(full)
                spans.append((len(c), len(full)))
                owner.append((ci, si))
        res = [[float("-inf")] * len(s) for s in suffixes]
        pad = self.tok.pad_token_id
        for s in range(0, len(seqs), batch_size):
            chunk = seqs[s:s + batch_size]
            L = max(map(len, chunk))
            inp = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for r, x in enumerate(chunk):
                inp[r, :len(x)] = torch.tensor(x)
                att[r, :len(x)] = 1
            inp, att = inp.to(self.device), att.to(self.device)
            with torch.inference_mode(), self._autocast():
                logits = self.model(input_ids=inp, attention_mask=att).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = inp[:, 1:]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            for r in range(len(chunk)):
                a, b = spans[s + r]
                val = float(tok_lp[r, a - 1:b - 1].sum())
                self.stats["score_tokens"] += b - a
                ci, si = owner[s + r]
                res[ci][si] = val
        return res

    def describe(self) -> dict:
        return {"model": self.spec.key, "hf_id": self.spec.hf_id, "init": self.init,
                "plan": self.plan.as_dict(), "stats": dict(self.stats)}


def trainable_summary(model) -> dict:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": tot, "params_trainable": tr}


__all__ = ["LLMEngine", "TrainCfg", "build_byte_tokenizer", "trainable_summary", "asdict"]
