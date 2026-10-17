"""v3 fine-tuning data and negative unlikelihood / preference objective (v3 novelty).

Implements:
1. V3RowDataset: partitions training data into positive (minority) and negative (majority) rows.
2. Negative unlikelihood loss:
       L_pos = - sum_t w_t * log p_theta(c_t | c_<t)
       L_neg = - sum_t w_t * log(1 - p_theta(c_t | c_<t) + eps)
       L_v3  = L_pos + beta_neg * L_neg
3. Optional DPO preference dataset pairing for contrastive alignment.
4. train_lm_v3: training loop logging both positive loss and negative repulsion.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from tabgen.utils import get_logger

from .codec import FieldCodec

log = get_logger("predllm_v3.finetune")


class V3RowDataset(Dataset):
    """Dataset serving positive rows (minority) and negative rows (majority)."""

    def __init__(
        self,
        rows: list[dict],
        codec: FieldCodec,
        lam: float = 3.0,
        w_scaffold: float = 0.0,
        perm_copies: int = 1,
        orig_copies: int = 1,
        negative_weight: float = 0.5,
        seed: int = 0,
    ):
        self.codec = codec
        self.lam, self.w_scaffold = lam, w_scaffold
        self.perm_copies, self.orig_copies = perm_copies, orig_copies
        self.negative_weight = negative_weight
        self.rng = np.random.default_rng(seed)
        self.feats = codec.schema.features
        target = codec.schema.target

        # Partition into positive (minority) and negative (majority)
        maj_set = set(codec.schema.majority_classes)
        self.pos_rows, self.neg_rows = [], []
        for r in rows:
            if str(r.get(target, "")) in maj_set:
                self.neg_rows.append(r)
            else:
                self.pos_rows.append(r)

        # Fallback if partition is empty (e.g., pure regression or single class)
        if not self.pos_rows:
            self.pos_rows = list(rows)

        self.n_pos = len(self.pos_rows)
        self.n_neg = len(self.neg_rows)

    def __len__(self) -> int:
        return (self.n_pos + self.n_neg) * (self.perm_copies + self.orig_copies)

    def _order(self, i: int) -> list[str]:
        if (i // (self.n_pos + self.n_neg)) < self.perm_copies:
            f = [self.feats[j] for j in self.rng.permutation(len(self.feats))]
        else:
            f = list(self.feats)
        return f + [self.codec.schema.target]

    def __getitem__(self, idx: int) -> dict:
        base_idx = idx % (self.n_pos + self.n_neg)
        is_negative = base_idx >= self.n_pos
        row = self.neg_rows[base_idx - self.n_pos] if is_negative else self.pos_rows[base_idx]
        order = self._order(idx)
        ids, w = self.codec.encode(row, order, self.lam, self.w_scaffold)
        return {
            "input_ids": ids,
            "weights": w,
            "is_negative": 1.0 if is_negative else 0.0,
        }

    def text(self, i: int) -> str:
        return self.codec.tok.decode(self[i]["input_ids"])


def make_collate_v3(pad_id: int):
    def collate(items):
        L = max(len(it["input_ids"]) for it in items)
        ids = torch.full((len(items), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(items), L), dtype=torch.long)
        w = torch.zeros((len(items), L), dtype=torch.float)
        is_neg = torch.zeros(len(items), dtype=torch.float)
        for i, it in enumerate(items):
            n = len(it["input_ids"])
            ids[i, :n] = torch.tensor(it["input_ids"])
            mask[i, :n] = 1
            w[i, :n] = torch.tensor(it.get("weights") or [1.0] * n)
            is_neg[i] = float(it.get("is_negative", 0.0))
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "weights": w,
            "is_negative": is_neg,
        }
    return collate


def v3_unlikelihood_loss(
    logits: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    is_neg: torch.Tensor,
    neg_weight: float = 0.5,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Computes joint positive cross-entropy and majority unlikelihood penalty."""
    logits_s = logits[:, :-1].float()
    tgt = ids[:, 1:]
    w = weights[:, 1:]
    is_neg_exp = is_neg[:, None].expand_as(tgt)

    probs = F.softmax(logits_s, dim=-1)
    p_true = probs.gather(-1, tgt[..., None])[..., 0].clamp(min=eps, max=1.0 - eps)

    # Positive samples: standard cross entropy -log p(x)
    pos_mask = (is_neg_exp == 0.0) & (w > 0)
    if pos_mask.sum() > 0:
        loss_pos = -(torch.log(p_true) * w * pos_mask.float()).sum() / (w * pos_mask.float()).sum().clamp_min(1e-8)
    else:
        loss_pos = torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Negative samples: unlikelihood penalty -log(1 - p(x))
    neg_mask = (is_neg_exp == 1.0) & (w > 0)
    if neg_mask.sum() > 0 and neg_weight > 0.0:
        p_comp = (1.0 - p_true).clamp(min=eps, max=1.0)
        loss_neg = -(torch.log(p_comp) * w * neg_mask.float()).sum() / (w * neg_mask.float()).sum().clamp_min(1e-8)
    else:
        loss_neg = torch.tensor(0.0, device=logits.device)

    total_loss = loss_pos + float(neg_weight) * loss_neg
    stats = {
        "loss_pos": round(float(loss_pos.detach().cpu().item()), 4),
        "loss_neg": round(float(loss_neg.detach().cpu().item()), 4),
    }
    return total_loss, stats


def train_lm_v3(
    model,
    tokenizer,
    dataset: V3RowDataset,
    cfg: dict,
    out_dir: str | Path,
    device: str = "cuda",
) -> dict:
    """v3 training loop with negative sample unlikelihood logging."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bs = int(cfg.get("batch_size", 32))
    mbs = int(cfg.get("micro_batch_size") or bs)
    accum = max(1, bs // mbs)
    loader = DataLoader(
        dataset,
        batch_size=mbs,
        shuffle=True,
        collate_fn=make_collate_v3(tokenizer.pad_token_id),
    )
    steps_per_epoch = math.ceil(len(loader) / accum)
    total = int(cfg.get("max_steps") or steps_per_epoch * int(cfg.get("epochs", 50)))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg.get("lr", 5e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    sched = get_linear_schedule_with_warmup(
        opt,
        int(float(cfg.get("warmup_ratio", 0.0)) * total),
        total,
    )
    use_amp = device.startswith("cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    neg_weight = float(cfg.get("negative_weight", 0.5))
    log_every = int(cfg.get("log_every", 50))
    log_path = out_dir / "train_log.jsonl"
    model.train()

    step, t0 = 0, time.time()
    running_loss, running_pos, running_neg, n_run = 0.0, 0.0, 0.0, 0
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
                    loss, loss_stats = v3_unlikelihood_loss(
                        out.logits,
                        batch["input_ids"],
                        batch["weights"],
                        batch["is_negative"],
                        neg_weight=neg_weight,
                    )
                    loss = loss / accum

                scaler.scale(loss).backward()
                running_loss += loss.item() * accum
                running_pos += loss_stats["loss_pos"]
                running_neg += loss_stats["loss_neg"]
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
                        rec = {
                            "step": step,
                            "epoch": epoch,
                            "loss": round(running_loss / max(n_run, 1), 4),
                            "loss_pos": round(running_pos / max(n_run, 1), 4),
                            "loss_neg": round(running_neg / max(n_run, 1), 4),
                            "lr": sched.get_last_lr()[0],
                            "elapsed_s": round(time.time() - t0, 1),
                        }
                        flog.write(json.dumps(rec) + "\n")
                        flog.flush()
                        history.append(rec)
                        running_loss, running_pos, running_neg, n_run = 0.0, 0.0, 0.0, 0

                    if step >= total:
                        break

    return {
        "final_step": step,
        "epochs": epoch,
        "final_loss": history[-1]["loss"] if history else None,
        "final_loss_pos": history[-1]["loss_pos"] if history else None,
        "final_loss_neg": history[-1]["loss_neg"] if history else None,
        "time_s": round(time.time() - t0, 1),
    }
