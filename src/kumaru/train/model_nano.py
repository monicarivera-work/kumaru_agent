"""A decoder-only transformer written from scratch, in ~250 readable lines.

This is not a toy architecture. It is the *same* design as Llama 3, Qwen 2.5,
Mistral, and Gemma — RMSNorm, rotary position embeddings, grouped-query
attention, SwiGLU feed-forward, pre-norm residuals — just with smaller numbers.
Scaling it to 7B is changing `n_layers`, `d_model`, and `n_heads`. Nothing
structural changes.

Reading this file once tells you what every large model is actually doing.

Why each choice, briefly
------------------------
**RMSNorm instead of LayerNorm.** Drops the mean-centering term and the bias.
Same training stability, ~10-15% less normalization compute. Universally adopted
since Llama.

**RoPE instead of learned positional embeddings.** Position is encoded by
*rotating* query and key vectors by an angle proportional to their index. Because
the attention score depends only on the rotation *difference*, the model sees
relative position directly, and the scheme extrapolates to longer sequences than
it was trained on.

**Grouped-query attention.** Queries get `n_heads`; keys and values share only
`n_kv_heads`. The KV cache is the dominant memory cost during generation, and
this divides it by `n_heads / n_kv_heads` with almost no quality loss. It is the
reason modern 7B models can serve 128k contexts.

**SwiGLU.** `(W1 x * sigmoid(W1 x)) . (W3 x)` then `W2`. The multiplicative gate
lets the network modulate its own activations. Empirically better than ReLU/GELU
at equal parameter count, which is why `d_ff ~= 2.67 x d_model` (three matrices)
rather than `4 x` (two).

**Pre-norm residuals.** Normalize *before* each sublayer, not after. This keeps a
clean identity path from input to output, so gradients reach early layers
without vanishing. It is what makes deep transformers trainable without careful
warmup tricks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class NanoConfig:
    """Architecture hyperparameters. Mirrors `train.config.NanoModelConfig`."""

    vocab_size: int
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    d_model: int = 512
    d_ff: int = 1408
    max_seq_length: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * weight."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Compute the statistic in fp32 even when activations are bf16: bf16 has
        # only ~3 decimal digits of precision, and squaring then averaging in it
        # loses enough accuracy to destabilize training.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin tables for rotary embeddings.

    Frequencies decay geometrically across dimension pairs: early pairs rotate
    fast (encoding fine local position), later pairs rotate slowly (encoding
    coarse global position). That multi-scale structure is what lets one scheme
    represent both "the previous word" and "4,000 tokens ago".
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, inv_freq)  # (seq, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate each adjacent dimension pair of `x` by its positional angle.

    `x` is (batch, heads, seq, head_dim). Treating dimensions in pairs as points
    in a plane, this is a standard 2-D rotation:
        x'_even = x_even·cos - x_odd·sin
        x'_odd  = x_even·sin + x_odd·cos
    """
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, : x.shape[-2], :]
    sin = sin[None, None, : x.shape[-2], :]
    rotated = torch.stack([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)
    return rotated.flatten(-2)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand shared KV heads to match the number of query heads."""
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


class Attention(nn.Module):
    """Causal self-attention with grouped-query heads."""

    def __init__(self, cfg: NanoConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        d, hd = cfg.d_model, cfg.head_dim

        # No biases: they contribute nothing measurable in transformers at scale
        # and cost parameters plus a kernel launch.
        self.q_proj = nn.Linear(d, cfg.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * hd, d, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, s, _ = x.shape
        cfg = self.cfg

        q = self.q_proj(x).view(b, s, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)

        # PyTorch dispatches this to FlashAttention when available: fused,
        # memory-linear in sequence length instead of quadratic. `is_causal`
        # applies the lower-triangular mask without materializing it.
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.o_proj(out.transpose(1, 2).reshape(b, s, cfg.d_model))


class SwiGLU(nn.Module):
    """Gated feed-forward network: `down(silu(gate(x)) * up(x))`."""

    def __init__(self, cfg: NanoConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """One transformer layer: attention, then feed-forward, both pre-normed."""

    def __init__(self, cfg: NanoConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # `x + f(norm(x))`: the residual stream is never normalized in place, so
        # there is an unobstructed gradient path from the loss to layer 0.
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class NanoLM(nn.Module):
    """A complete causal language model."""

    def __init__(self, cfg: NanoConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Input and output embeddings are the same matrix. Saves
            # vocab_size x d_model parameters — for a small model with a large
            # vocabulary that can be half the total — and acts as a regularizer.
            self.lm_head.weight = self.embed.weight

        self.register_buffer("_cos", torch.empty(0), persistent=False)
        self.register_buffer("_sin", torch.empty(0), persistent=False)

        self.apply(self._init_weights)
        # Scaled init for residual projections: with L layers summing into the
        # residual stream, variance grows like L. Scaling by 1/sqrt(2L) keeps
        # activations well-conditioned at initialization in deep models.
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._cos.numel() == 0 or self._cos.shape[0] < seq_len or self._cos.device != device:
            cos, sin = build_rope_cache(
                max(seq_len, self.cfg.max_seq_length), self.cfg.head_dim, self.cfg.rope_theta, device
            )
            self._cos, self._sin = cos.to(dtype), sin.to(dtype)
        return self._cos[:seq_len], self._sin[:seq_len]

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        _, seq_len = input_ids.shape
        x = self.embed(input_ids)
        cos, sin = self._rope(seq_len, x.device, x.dtype)

        for block in self.blocks:
            x = block(x, cos, sin)

        logits = self.lm_head(self.norm(x))
        out: dict[str, Tensor] = {"logits": logits}

        if labels is not None:
            # Labels arrive pre-shifted from PackedTokenDataset, so no shifting
            # here. `ignore_index=-100` skips padded/masked positions.
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return out

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        return sum(p.numel() for p in params)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Autoregressive sampling.

        Deliberately naive: it re-runs the full forward pass for every token
        instead of caching keys and values, making it O(n^2) overall. A KV cache
        is the single most important inference optimization, and leaving it out
        here keeps the mechanism obvious. Production serving uses the
        `transformers` or llama.cpp backends, which implement it properly.
        """
        self.eval()
        for _ in range(max_new_tokens):
            window = input_ids[:, -self.cfg.max_seq_length :]
            logits = self(window)["logits"][:, -1, :]

            if temperature <= 0:
                next_token = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return input_ids


def estimate_parameters(cfg: NanoConfig) -> dict[str, int]:
    """Parameter count by component, without building the model.

    Useful for sizing a run before you spend GPU-hours on it, and for sanity
    checks against the scaling laws in docs/SCALABILITY.md.
    """
    hd = cfg.head_dim
    embed = cfg.vocab_size * cfg.d_model
    attn_per_layer = (
        cfg.d_model * cfg.n_heads * hd      # q
        + cfg.d_model * cfg.n_kv_heads * hd  # k
        + cfg.d_model * cfg.n_kv_heads * hd  # v
        + cfg.n_heads * hd * cfg.d_model     # o
    )
    ffn_per_layer = 3 * cfg.d_model * cfg.d_ff
    norms_per_layer = 2 * cfg.d_model

    blocks = cfg.n_layers * (attn_per_layer + ffn_per_layer + norms_per_layer)
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model

    return {
        "embedding": embed,
        "blocks": blocks,
        "lm_head": head,
        "total": embed + blocks + head + cfg.d_model,
    }
