# LLM switch

Every command in v1 and v2 accepts `--llm <alias|key|hf_id>` and `--mode auto|full|lora|qlora`.
The registry is `configs/llms.yaml` and the resolver is `tabgen/llm/registry.py`.
Model loading (full / LoRA / 4-bit QLoRA) happens in one place: `tabgen/llm/loader.py`.

```bash
python -m tabgen llm list                      # all models, and which ones fit your GPU
python -m tabgen llm check --llm 1.5b          # load + 1 training step + 1 generation, prints peak memory
python -m predllm_v1 run --dataset adult --llm 3b      # Qwen2.5-3B with QLoRA
python -m predllm_v2 run --dataset adult --llm 8b      # Qwen3-8B with QLoRA (needs >= 12 GB)
```

## Compatible models (fine-tuning is involved, so only these are supported)

| alias | key | HF id | params | default mode | min VRAM | gated |
|---|---|---|---|---|---|---|
| tiny | tiny-gpt2 | sshleifer/tiny-gpt2 | 0.1M | full | 0 | no (tests only) |
| | tiny-llama | hf-internal-testing/tiny-random-LlamaForCausalLM | 1M | lora | 0 | no (tests only) |
| paper | distilgpt2 | distilbert/distilgpt2 | 82M | full | 2 GB | no |
| | gpt2 | openai-community/gpt2 | 124M | full | 3 GB | no |
| | gpt2-medium | openai-community/gpt2-medium | 355M | full | 6 GB | no |
| | smollm2-360m | HuggingFaceTB/SmolLM2-360M | 360M | full | 6 GB | no |
| 0.5b | qwen2.5-0.5b | Qwen/Qwen2.5-0.5B | 0.5B | lora | 3 GB | no |
| | qwen3-0.6b | Qwen/Qwen3-0.6B-Base | 0.6B | lora | 3 GB | no |
| 1b | llama3.2-1b | meta-llama/Llama-3.2-1B | 1.2B | lora | 4 GB | yes |
| 1.5b | qwen2.5-1.5b | Qwen/Qwen2.5-1.5B | 1.5B | lora | 5 GB | no |
| 1.7b | qwen3-1.7b | Qwen/Qwen3-1.7B-Base | 1.7B | lora | 6 GB | no |
| | smollm2-1.7b | HuggingFaceTB/SmolLM2-1.7B | 1.7B | lora | 6 GB | no |
| 3b | qwen2.5-3b | Qwen/Qwen2.5-3B | 3.1B | qlora | 5 GB | no |
| | llama3.2-3b | meta-llama/Llama-3.2-3B | 3.2B | qlora | 5 GB | yes |
| 4b | qwen3-4b | Qwen/Qwen3-4B-Base | 4.0B | qlora | 6 GB | no |
| 7b | qwen2.5-7b | Qwen/Qwen2.5-7B | 7.6B | qlora | 10 GB | no |
| | llama3.1-8b | meta-llama/Llama-3.1-8B | 8.0B | qlora | 12 GB | yes |
| 8b | qwen3-8b | Qwen/Qwen3-8B-Base | 8.2B | qlora | 12 GB | no |

**Why these, and only these.**
* All are **base** causal LMs that `AutoModelForCausalLM` can fine-tune, and PEFT LoRA with
  `target_modules="all-linear"` works on them (GPT-2's Conv1D included).
* All use **byte-level BPE** tokenizers in which `,` is a single token. v2's schema decoder
  tokenizes keys, values and separators separately and needs this property.
  SentencePiece models (Llama-2, Mistral) add a dummy-prefix space to every piece, so they are left
  out on purpose.
* Chat/instruct checkpoints are left out: the method fine-tunes on raw serialized rows, and chat
  templates would only add noise.

**Gated models.** Accept the licence on the model's Hugging Face page, then run `huggingface-cli login`.

**Modes.**
* `full` trains every weight in fp32 with bf16 autocast (the paper's setting).
* `lora` loads bf16 base weights and trains r=16 adapters. The adapter is saved (a few MB), and LR is
  `finetune.lr_lora` = 2e-4 with micro-batch 8.
* `qlora` loads 4-bit NF4 base weights with double quantisation and gradient checkpointing, and
  trains LoRA adapters with micro-batch 4. It needs bitsandbytes and a CUDA GPU.

The effective batch size stays 32 (paper) through gradient accumulation.

**On a 6 GB laptop GPU (e.g. RTX 4050):** distilgpt2 / gpt2 in full mode, up to 1.5B with LoRA,
and 3B-4B with QLoRA. 7B-8B need a larger GPU (or a cloud box). The code is identical there.

**Unregistered models.** `--llm some-org/model` is accepted, defaults to LoRA and prints a warning.
Only the models above were checked for compatibility.
