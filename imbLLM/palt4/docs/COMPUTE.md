# Compute: laptop and H100

Two machines, one code path:
- **RTX 4050 Laptop (6 GB, Ada, bf16):** always available.
- **H100 80 GB:** about 20 h/week, in sessions.

`--hardware auto | rtx4050_6gb | h100_80gb` picks the device profile. The planner then chooses full / LoRA / QLoRA, precision, micro-batch and gradient accumulation (effective batch 32), or refuses up front with a reason.

## The model switch: Qwen3 Base ladder

Checked 28 Sep 2026: [V] = verified on an official page, [E] = estimate.

| key | HF id | params | laptop | H100 |
|---|---|---|---|---|
| `qwen3-0.6b` | Qwen/Qwen3-0.6B-Base | 0.6B | LoRA ✓ (development) | ✓ |
| `qwen3-1.7b` | Qwen/Qwen3-1.7B-Base | 1.7B | QLoRA ✓ (**thesis backbone**) | LoRA bf16 ✓ |
| `qwen3-4b` | Qwen/Qwen3-4B-Base | 4.0B | QLoRA, tight | ✓ |
| `qwen3-8b` | Qwen/Qwen3-8B-Base | 8.2B | ✗ | ✓ |
| `qwen3-14b` | Qwen/Qwen3-14B-Base | 14.8B | ✗ | ✓ |
| `qwen2.5-32b` | Qwen/Qwen2.5-32B | 32.5B | ✗ | QLoRA ~24 GB / LoRA ~65 GB |

All use byte-level BPE tokenizers. Qwen splits every digit into its own token, so numbers cost one token per digit.
- Qwen3 needs transformers ≥ 4.51 [V].
- All are Apache-2.0 [V], except Qwen2.5-3B (Qwen Research licence).
- There is **no Qwen3-32B-Base** [V], so the 32B rung is Qwen2.5.

**Rejected by `tilt models`, with the reason printed:**
- **Qwen3.5 / 3.6:**
  - hybrid linear-attention layers keep a recurrent state, so there is no KV cache to fork for candidate scoring;
  - the fast kernels are Linux-only;
  - prefix caching does nothing on rows under 528 tokens [V].
- **MoE (30B-A3B, 35B-A3B):**
  - 4-bit QLoRA does not quantise fused experts under transformers v5 [V, bnb #1849];
  - PEFT `all-linear` skips the expert MLPs [V, peft #3801].

## Memory at 256 tokens, micro-batch 4–8, gradient checkpointing, LoRA r = 16 [E]

| size | bf16 inference | bf16 LoRA | QLoRA |
|---|---|---|---|
| 0.5–0.8B | 1.5–2.5 GB | 3–5 GB | 2.5–4 GB |
| 1.5–2B | 4–5 GB | 5.5–7.5 GB | 3–4.5 GB |
| 3–4B | 7–9 GB | 10–12 GB | 4–6 GB |
| 7–9B | 16–19 GB | 18–24 GB | 7–10 GB |
| 14B | ~30 GB | 33–38 GB | 12–15 GB |
| 32B | 55–66 GB | 60–72 GB | 22–26 GB |

**Watch the logits.** A 152k vocabulary × 2,048 tokens in fp32, with its gradient, is about 2.5 GB. On the laptop that is most of the memory. Keep sequences short (rows are about 40–120 tokens) and micro-batches small.

## Throughput and the weekly budget

| setting | speed | source |
|---|---|---|
| RTX 4060 (8 GB), Qwen2.5-1.5B LoRA/QLoRA | 500–628 tokens/s | [V, 2509.12229] |
| RTX 4050, same setup | about 300–600 tokens/s | [E] |
| H100, bf16 LoRA | 0.6B 30–50k · 1.7B 15–30k · 8B 4–8k · 14B 2–4k · 32B 1–2k tokens/s | [E, 6N FLOPs at 25–40% MFU] |

**One fine-tune** of Qwen3-1.7B on a natural table: about 5k–45k corpus rows × ~80 tokens × 10 epochs ≈ 4–36M tokens, i.e. **3–40 minutes on the H100**. Generation (|largest class| × pool ×4) costs about as much again, so budget sampling separately.

**A 20 h/week plan:**

| week | where | what |
|---|---|---|
| 1 | laptop | `tilt oracle` (CPU) + `tilt smoke` + `laptop_dev` on natural_fast. **Go/no-go on E0.** |
| 2–3 | H100 | `thesis_main --config h100_main` on the 7 small tables, 10 seeds. Use `--resume` across sessions. |
| 4 | H100 | the 3 large tables (bank_full, mammography, satellite), 5 seeds |
| 5 | H100 | `utility` + `budget` + `validity`. These are sampling-only, so the cached fine-tunes are reused and they are cheap. |
| 6 | H100 | `sampler`; then `scale` on natural_fast, 3 seeds (8B, 14B, 32B are the expensive rungs) |
| 7 | either | `scarcity`, `replication`, contamination probes, reports and figures |

**Session hygiene on the H100:**
- `tilt pull --model qwen3-1.7b` once, to cache the weights.
- Run `tilt run <exp> --config h100_main --resume` in a `tmux` session.
- For several GPUs, use `--shard i/N`, one process per GPU.
- Everything writes to `runs/`; copy that folder back to the laptop for `tilt report`.

## Inference

The reference implementation is HF transformers (KV cache; batched `generate`; `score` for candidate sets) on both machines.
- **vLLM** is only useful on the H100, under Linux, for bulk unguided sampling with LoRA adapters.
- It gives no KV-cache access, and it is not supported on Windows [V].
- TILT's SIR can use it for the pool; SMC cannot.
