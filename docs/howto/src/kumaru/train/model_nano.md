# `src\kumaru\train\model_nano.py`

> A readable decoder-only transformer implementation used by Kumaru's from-scratch NanoLM pretraining path.

**Read this when:** you want to understand the architecture being pretrained, change NanoLM size, add an architectural experiment, or estimate parameter counts before running.

---

## What it does

This file implements a causal language model in PyTorch: token embeddings, RMSNorm, rotary position embeddings, grouped-query attention, SwiGLU feed-forward layers, pre-norm residual blocks, tied output embeddings, loss computation, and simple autoregressive sampling.

The model is intentionally small by default, but structurally it mirrors modern Llama/Qwen/Mistral/Gemma-style decoders. Scaling means changing dimensions and layer counts, not replacing the architecture.

## Why it exists

Framework trainers hide the details that matter when a run diverges. `model_nano.py` keeps the transformer mechanics local and readable so operators can connect config changes to parameter counts, memory use, and generated samples.

Without this file, Kumaru's `pretrain` task would need an external model class and would no longer demonstrate RMSNorm, RoPE, GQA, SwiGLU, initialization, and loss shifting end to end.

## Mental model

1. **A block is pre-norm attention plus pre-norm SwiGLU.** Residual paths carry gradients cleanly.
2. **RoPE modifies queries and keys, not token embeddings.** Position affects attention scores through rotations.
3. **GQA shares key/value heads.** `n_heads / n_kv_heads` is the KV sharing factor.

## Public API

| Name | Signature | What it's for |
|---|---|---|
| `NanoConfig` | `@dataclass class NanoConfig` | Architecture hyperparameters for `NanoLM`. |
| `NanoConfig.__post_init__` | `__post_init__() -> None` | Validates `d_model % n_heads == 0` and `n_heads % n_kv_heads == 0`. |
| `NanoConfig.head_dim` | `property -> int` | Attention head width, `d_model // n_heads`. |
| `RMSNorm` | `RMSNorm(dim: int, eps: float = 1e-6)` | Root-mean-square normalization module. |
| `RMSNorm.forward` | `forward(x: Tensor) -> Tensor` | Normalizes in fp32 and returns original dtype. |
| `build_rope_cache` | `build_rope_cache(seq_len: int, head_dim: int, theta: float, device: torch.device) -> tuple[Tensor, Tensor]` | Precomputes cosine/sine tables for RoPE. |
| `apply_rope` | `apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor` | Rotates adjacent query/key dimensions by position. |
| `repeat_kv` | `repeat_kv(x: Tensor, n_rep: int) -> Tensor` | Expands shared KV heads to match query heads. |
| `Attention` | `Attention(cfg: NanoConfig)` | Causal self-attention with grouped-query heads. |
| `Attention.forward` | `forward(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor` | Applies q/k/v projections, RoPE, GQA expansion, causal SDPA, output projection. |
| `SwiGLU` | `SwiGLU(cfg: NanoConfig)` | Gated feed-forward network. |
| `SwiGLU.forward` | `forward(x: Tensor) -> Tensor` | Computes `down(silu(gate(x)) * up(x))`. |
| `Block` | `Block(cfg: NanoConfig)` | One transformer layer. |
| `Block.forward` | `forward(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor` | Applies residual attention and residual feed-forward. |
| `NanoLM` | `NanoLM(cfg: NanoConfig)` | Complete causal LM. |
| `NanoLM.forward` | `forward(input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]` | Returns logits and optional cross-entropy loss. |
| `NanoLM.num_parameters` | `num_parameters(trainable_only: bool = False) -> int` | Counts total or trainable parameters. |
| `NanoLM.generate` | `generate(input_ids: Tensor, max_new_tokens: int = 128, temperature: float = 0.8, top_k: int = 50, eos_token_id: int | None = None) -> Tensor` | Simple autoregressive sampling without a KV cache. |
| `estimate_parameters` | `estimate_parameters(cfg: NanoConfig) -> dict[str, int]` | Estimates embedding, block, lm_head, and total parameter counts. |

## How to use it

Estimate the starter model:

```python
from transformers import AutoTokenizer
from kumaru.train.model_nano import NanoConfig, estimate_parameters

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
cfg = NanoConfig(vocab_size=len(tok), n_layers=8, n_heads=8, n_kv_heads=4, d_model=512, d_ff=1408)
print(estimate_parameters(cfg))
```

Instantiate and run a forward pass:

```python
import torch
from kumaru.train.model_nano import NanoConfig, NanoLM

cfg = NanoConfig(vocab_size=32000, max_seq_length=128)
model = NanoLM(cfg)
ids = torch.randint(0, cfg.vocab_size, (2, 128))
out = model(ids, labels=ids)
print(out["logits"].shape, out["loss"].item())
```

Use the checked-in pretraining path:

```powershell
kumaru train configs\train\pretrain-nano.yaml --dry-run
kumaru train configs\train\pretrain-nano.yaml
```

The matching YAML architecture section is:

```yaml
model:
  vocab_size: 0
  n_layers: 8
  n_heads: 8
  n_kv_heads: 4
  d_model: 512
  d_ff: 1408
  max_seq_length: 512
  dropout: 0.0
  rope_theta: 10000.0
  tie_embeddings: true
```

## Tuning / operational notes

The default config describes a roughly 101M-parameter model; with the large Qwen tokenizer vocabulary, about 78M of that is the tied embedding table. It can train to coherent local text in under an hour on one RTX 3090 when paired with the sample-scale pretraining loop, but it will not become a useful assistant.

Architecture tradeoffs:

- `d_model` and `n_layers` dominate quality and compute.
- `n_heads` must divide `d_model`; `head_dim = d_model / n_heads`.
- `n_kv_heads` must divide `n_heads`. Fewer KV heads reduce generation KV-cache memory by `n_heads / n_kv_heads`.
- `d_ff` should stay near the SwiGLU ratio, about 2.67-2.75x `d_model`.
- `tie_embeddings: true` saves `vocab_size * d_model` parameters, which matters with a 152k-token tokenizer.
- `generate()` is deliberately naive and O(n²) because it does not cache K/V states.

If you hit OOM, change these in order:

1. Lower `optim.per_device_batch_size` in `pretrain-nano.yaml`.
2. Lower `model.max_seq_length`.
3. Lower `model.n_layers`.
4. Lower `model.d_model` and adjust `n_heads` so divisibility holds.
5. Keep `tie_embeddings: true`.
6. Move to distributed/sharded training before attempting truly large models.

## How to extend it

To scale the model up modestly, keep the same relationships:

```yaml
model:
  vocab_size: 0
  n_layers: 16
  n_heads: 12
  n_kv_heads: 4
  d_model: 768
  d_ff: 2048
  max_seq_length: 1024
  dropout: 0.0
  rope_theta: 10000.0
  tie_embeddings: true
```

To add a different normalization layer, implement a module with the same `forward(x)` shape contract and replace `RMSNorm` in `Block` and `NanoLM`. To add MoE, replace or wrap `SwiGLU` inside `Block`, preserve input/output shape `(batch, seq, d_model)`, and update `estimate_parameters()` to include expert matrices.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ValueError: d_model must be divisible by n_heads` | Invalid architecture config | Choose `n_heads` that divides `d_model`. |
| `ValueError: n_heads must be divisible by n_kv_heads` | Invalid GQA grouping | Choose `n_kv_heads` as a divisor of `n_heads`. |
| NaN loss early in pretraining | LR too high, bad precision, or no warmup | Lower LR, keep warmup, check bf16 support. |
| Generation is slow | `generate()` has no KV cache and reruns the full context | Use it only for samples; production serving uses other backends. |
| Model overfits instantly | Corpus is tiny relative to parameter count | Use more text; about 20 tokens per parameter is the sizing target. |
| OOM after increasing context | Attention/activation memory grew with sequence length | Reduce `max_seq_length` or batch size. |

## Related files

- `src\kumaru\train\pretrain_nano.py` — constructs `NanoConfig`, trains `NanoLM`, and saves checkpoints.
- `src\kumaru\train\config.py` — defines `NanoModelConfig`, the YAML mirror of `NanoConfig`.
- `configs\train\pretrain-nano.yaml` — checked-in architecture values.
- `src\kumaru\train\data.py` — supplies shifted packed-token batches consumed by `NanoLM.forward`.
- `docs\SCALABILITY.md` — referenced by parameter-estimation comments for scaling context.
