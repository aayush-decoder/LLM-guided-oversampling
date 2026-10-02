# Shared fine-tuning loop for both versions.

# Objective (Eq. 1 of the paper, generalised with per-token weights w_t):

#     L(theta) = - sum_t w_t * log p_theta(c_t | c_<t)  /  sum_t w_t

# v1 uses w_t = 1 on every real token (plain autoregressive NLL = the paper).
# v2 passes value-/target-weighted w_t (see v2/docs/NOVELTY.md, section N3).

# Items yielded by the dataset: {"input_ids": list[int], "weights": list[float]}.

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..utils import get_logger

log = get_logger("tabgen.train")


def make_collate(pad_id: int):
    def collate(items):
        L = max(len(it["input_ids"]) for it in items)
        ids = torch.full((len(items), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(items), L), dtype=torch.long)
        w = torch.zeros((len(items), L), dtype=torch.float)
        for i, it in enumerate(items):
            n = len(it["input_ids"])
            ids[i, :n] = torch.tensor(it["input_ids"])
            mask[i, :n] = 1
            w[i, :n] = torch.tensor(it.get("weights") or [1.0] * n)
        return {"input_ids": ids, "attention_mask": mask, "weights": w}
    return collate


def weighted_lm_loss(logits: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    logits = logits[:, :-1].float()
    tgt = ids[:, 1:]
    w = weights[:, 1:]
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
    return (nll * w.reshape(-1)).sum() / w.sum().clamp_min(1e-8)


def train_lm(model, tokenizer, dataset, cfg: dict, out_dir: str | Path, device: str = "cuda") -> dict:
    """cfg keys: epochs, batch_size, micro_batch_size, lr, weight_decay, warmup_ratio, max_grad_norm,
    max_steps (optional, overrides epochs), log_every."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bs = int(cfg.get("batch_size", 32))
    mbs = int(cfg.get("micro_batch_size") or bs)
    accum = max(1, bs // mbs)
    loader = DataLoader(dataset, batch_size=mbs, shuffle=True, collate_fn=make_collate(tokenizer.pad_token_id))
    steps_per_epoch = math.ceil(len(loader) / accum)
    total = int(cfg.get("max_steps") or steps_per_epoch * int(cfg.get("epochs", 50)))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 5e-5)), weight_decay=float(cfg.get("weight_decay", 0.0)))
    sched = get_linear_schedule_with_warmup(opt, int(float(cfg.get("warmup_ratio", 0.0)) * total), total)
    use_amp = device.startswith("cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    log_every = int(cfg.get("log_every", 50))
    log_path = out_dir / "train_log.jsonl"
    model.train()
    step, t0, running, n_run = 0, time.time(), 0.0, 0
    history = []
    with open(log_path, "w", encoding="utf-8") as flog:
        epoch = 0
        while step < total:
            epoch += 1
            opt.zero_grad(set_to_none=True)
            for i, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = weighted_lm_loss(out.logits, batch["input_ids"], batch["weights"]) / accum
                scaler.scale(loss).backward()
                running += loss.item() * accum
                n_run += 1
                if (i + 1) % accum == 0 or (i + 1) == len(loader):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % log_every == 0 or step == total:
                        rec = {"step": step, "epoch": epoch, "loss": running / max(n_run, 1),
                               "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
                        flog.write(json.dumps(rec) + "\n")
                        flog.flush()
                        history.append(rec)
                        log.info(f"step {step}/{total} ep {epoch} loss {rec['loss']:.4f}")
                        running, n_run = 0.0, 0
                    if step >= total:
                        break
    model.eval()
    return {"steps": step, "epochs_run": epoch, "final_loss": history[-1]["loss"] if history else None,
            "train_seconds": round(time.time() - t0, 1), "n_sequences": len(dataset)}
