"""Trainable plain-PyTorch ZONOS3-Mini language model.

The parameter names and tensor shapes are those of the public release checkpoint (``model.pth`` of
``hf://Zyphra/ZONOS3-Mini/lm/zonos3_mini``), so the checkpoint loads with a strict ``load_state_dict`` and a trained
model exports to the same format without any key mapping. The forward computes the training-time arithmetic of the
ZONOS3 Mini architecture:

- a sum of per-column token embeddings; the reserved speaker row and the condition rows replace their summed
  embeddings (speaker projection, condition bucket tables) before the weightless embedding RMSNorm;
- Full AttnRes: every sublayer reads a softmax-weighted mix of all earlier sources (the embedding plus the output of
  every attention/GDN and FFN/MoE sublayer), selected by a learned query per sublayer;
- gated DeltaNet (GDN) mixers on three of every four layers (FLA's chunked Triton kernels, the kernels the model was
  trained with) and gated global attention with QK norm and interleaved RoPE on the others;
- a dense SwiGLU FFN on layer 0 and a latent MoE (64 experts, top-2, 256-wide latent) on the others;
- the 10-codebook output head with the heterogeneous audio loss (ordinary codes + EOA per codebook, the semantic
  codebook weighted) over packed sequences.

Packed training only: ``token_values`` is ``[tokens, n_codebooks + 1]`` with ``cu_seqlens`` document boundaries.
Without FLA (``gdn_backend="torch"``) the GDN runs a slow token-by-token recurrence, for CPU tests and parity checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

GDN_HEADS, GDN_HEAD_DIM, GDN_KERNEL = 4, 128, 4
GDN_INNER = GDN_HEADS * GDN_HEAD_DIM


def _round_up(value: float, multiple: int) -> int:
    return multiple * ((int(value) + multiple - 1) // multiple)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


@torch.no_grad()
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    initialized = tensor.float()
    nn.init.trunc_normal_(initialized, mean=mean, std=std, a=a, b=b)
    tensor.copy_(initialized)
    return tensor


@dataclass
class Zonos3Args:
    n_layers: int = 11
    dim: int = 1024
    head_dim: int = 32
    n_heads: int = 16
    n_kv_heads: int = 4
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seqlen: int = 32768
    loss_softcap: float = 15.0
    gdn_period: int = 4  # layers with (layer + 1) % gdn_period == 0 use global attention
    ffn_hidden: int = 2048
    moe_hidden: int = 1536
    moe_routed_dim: int = 256
    n_experts: int = 64
    moe_topk: int = 2
    n_codebooks: int = 10
    codebook_size: int = 4096
    eoa_id: int = 4096
    audio_pad_id: int = 4097
    text_vocab: int = 8196
    text_extra_tokens: int = 0
    speaker_embedding_dim: int = 192
    condition_bucket_counts: tuple[int, ...] = (10, 10, 30)
    semantic_codebook_loss_weight: float = 3.0
    gdn_backend: str = "auto"  # auto | fla | torch
    init_base_std: float = 0.02

    @property
    def audio_vocab(self) -> int:
        return self.codebook_size + 2

    @property
    def text_rows(self) -> int:
        return self.text_vocab + 1 + self.text_extra_tokens

    @property
    def n_sources(self) -> int:
        return 2 * self.n_layers + 1

    def uses_gdn(self, layer: int) -> bool:
        return (layer + 1) % self.gdn_period != 0

    @classmethod
    def from_params(cls, params: dict, **overrides) -> "Zonos3Args":
        """Derive the module sizes from a release ``params.json`` (the whole document or its ``model`` table)."""
        p = params.get("model", params)
        required = {
            "residual_type": "full_attnres",
            "speaker_enabled": True,
            "condition_tokens_enabled": True,
            "moe_router_type": "linear",
            "moe_router_use_eda": False,
            "moe_start_from_layer": 1,
            "moe_end_from_layer": 0,
        }
        for key, wanted in required.items():
            if p.get(key) != wanted:
                raise ValueError(f"unsupported checkpoint: {key}={p.get(key)!r} (expected {wanted!r})")
        pattern = str(p.get("attention_pattern"))
        if not pattern.startswith("gdn") or not pattern.endswith("_global1"):
            raise ValueError(f"unsupported attention_pattern {pattern!r}")
        gdn_period = int(pattern[len("gdn") : -len("_global1")]) + 1
        if p.get("span_end_id") is not None:
            raise ValueError("speech-edit checkpoints (span_end_id) are not supported by this trainer")
        dim = int(p["dim"])
        multiple_of = int(p.get("multiple_of", 256))
        moe_multiple_of = int(p.get("moe_multiple_of") or multiple_of)
        ffn_mult = float(p["ffn_dim_multiplier"])
        moe_mult = float(p.get("moe_ffn_dim_multiplier") or ffn_mult)
        count = int(p.get("condition_count", 3))
        counts = p.get("condition_bucket_counts") or [int(p.get("condition_n_buckets", 10))] * count
        if len(counts) != count:
            raise ValueError("condition_bucket_counts does not match condition_count")
        if int(p["eoa_id"]) != int(p["codebook_size"]) or int(p["audio_pad_id"]) != int(p["codebook_size"]) + 1:
            raise ValueError("unsupported eoa_id / audio_pad_id")
        args = cls(
            n_layers=int(p["n_layers"]),
            dim=dim,
            head_dim=int(p["head_dim"]),
            n_heads=int(p["n_heads"]),
            n_kv_heads=int(p["n_kv_heads"]),
            norm_eps=float(p["norm_eps"]),
            rope_theta=float(p["rope_theta"]),
            max_seqlen=int(p["max_seqlen"]),
            loss_softcap=float(p["loss_softcap"]),
            gdn_period=gdn_period,
            ffn_hidden=_round_up(ffn_mult * dim, multiple_of),
            moe_hidden=_round_up(moe_mult * dim, moe_multiple_of),
            moe_routed_dim=max(1, int(round(dim * float(p["moe_alpha"])))),
            n_experts=int(p["moe_n_experts"]),
            moe_topk=int(p["moe_router_topk"]),
            n_codebooks=int(p["n_codebooks"]),
            codebook_size=int(p["codebook_size"]),
            eoa_id=int(p["eoa_id"]),
            audio_pad_id=int(p["audio_pad_id"]),
            text_vocab=int(p["text_vocab"]),
            text_extra_tokens=int(p.get("text_extra_tokens", 0) or 0),
            speaker_embedding_dim=int(p["speaker_embedding_dim"]),
            condition_bucket_counts=tuple(int(c) for c in counts),
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args


# --------------------------------------------------------------------------------------------------- helpers


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return cap * torch.tanh(x / cap)


def sequence_positions(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Position of every packed token inside its own sequence."""
    total = int(cu_seqlens[-1])
    positions = torch.arange(total, device=cu_seqlens.device)
    seq_idx = torch.bucketize(positions, cu_seqlens, right=True) - 1
    return positions - cu_seqlens[seq_idx]


def rope_interleaved(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Interleaved rotary embedding ([..., 2i], [..., 2i+1] pairs) in float32, as the release runtime."""
    hd = x.shape[-1]
    frequencies = theta ** (-torch.arange(0, hd, 2, dtype=torch.float32, device=x.device) / hd)
    angles = positions.float()[:, None] * frequencies[None, :]
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
    pairs = x.float().reshape(*x.shape[:-1], hd // 2, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2).to(x.dtype)


class _Weight(nn.Module):
    """A bare weight tensor (``wkv`` [2, kv, dim], ``w_in`` [2, hidden, dim]) with an optional LoRA branch."""

    def __init__(self, *shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*shape))
        self.register_parameter("lora_a", None)
        self.register_parameter("lora_b", None)
        self.lora_dropout = nn.Identity()
        self.lora_scale = 1.0

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[-1])

    @property
    def out_features(self) -> int:
        return int(self.weight.numel() // self.weight.shape[-1])

    def enable_lora(self, rank: int, alpha: float, dropout: float) -> None:
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if self.lora_a is not None or self.lora_b is not None:
            raise RuntimeError("LoRA is already enabled for this module")
        self.weight.requires_grad_(False)
        self.lora_a = nn.Parameter(self.weight.new_empty(rank, self.in_features))
        self.lora_b = nn.Parameter(self.weight.new_zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_scale = float(alpha) / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = _local(self.weight).reshape(self.out_features, self.in_features)
        base = F.linear(x, weight)
        if self.lora_a is None:
            return base
        adapter = F.linear(F.linear(self.lora_dropout(x), _local(self.lora_a)), _local(self.lora_b))
        return base + adapter * self.lora_scale


class _Experts(nn.Module):
    """Per-expert weights [E, out, in] with an optional per-expert LoRA branch ([E, r, in], [E, out, r])."""

    def __init__(self, n_experts: int, moe_hidden: int, routed_dim: int):
        super().__init__()
        # gate_up_proj rows [0, moe_hidden) are the gate, [moe_hidden, 2 moe_hidden) the up projection.
        self.gate_up_proj = nn.Parameter(torch.empty(n_experts, 2 * moe_hidden, routed_dim))
        self.down_proj = nn.Parameter(torch.empty(n_experts, routed_dim, moe_hidden))
        for name in ("gate_up_proj_lora_a", "gate_up_proj_lora_b", "down_proj_lora_a", "down_proj_lora_b"):
            self.register_parameter(name, None)
        self.lora_scale = 1.0
        self.moe_hidden = moe_hidden

    def enable_lora(self, rank: int, alpha: float) -> None:
        if self.gate_up_proj_lora_a is not None:
            raise RuntimeError("LoRA is already enabled for the experts")
        self.gate_up_proj.requires_grad_(False)
        self.down_proj.requires_grad_(False)
        e, out_gu, routed = self.gate_up_proj.shape
        _, out_d, hidden = self.down_proj.shape
        self.gate_up_proj_lora_a = nn.Parameter(self.gate_up_proj.new_empty(e, rank, routed))
        self.gate_up_proj_lora_b = nn.Parameter(self.gate_up_proj.new_zeros(e, out_gu, rank))
        self.down_proj_lora_a = nn.Parameter(self.down_proj.new_empty(e, rank, hidden))
        self.down_proj_lora_b = nn.Parameter(self.down_proj.new_zeros(e, out_d, rank))
        for a in (self.gate_up_proj_lora_a, self.down_proj_lora_a):
            for expert in range(e):
                nn.init.kaiming_uniform_(a[expert], a=math.sqrt(5))
        self.lora_scale = float(alpha) / rank

    def expert_weights(self, expert: int) -> tuple[torch.Tensor, torch.Tensor]:
        gate_up = _local(self.gate_up_proj)[expert]
        down = _local(self.down_proj)[expert]
        if self.gate_up_proj_lora_a is not None:
            gate_up = gate_up + self.lora_scale * (
                _local(self.gate_up_proj_lora_b)[expert] @ _local(self.gate_up_proj_lora_a)[expert]
            ).to(gate_up.dtype)
            down = down + self.lora_scale * (
                _local(self.down_proj_lora_b)[expert] @ _local(self.down_proj_lora_a)[expert]
            ).to(down.dtype)
        return gate_up, down

    @torch.compiler.disable
    def forward(self, latent_sorted: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        """Experts over tokens sorted by expert: ``latent_sorted`` [sum counts, routed], counts [E]."""
        splits = torch.split(latent_sorted, tokens_per_expert.tolist(), dim=0)
        outputs = []
        for expert, chunk in enumerate(splits):
            if chunk.shape[0] == 0:
                continue
            gate_up_w, down_w = self.expert_weights(expert)
            gate_up = F.linear(chunk, gate_up_w)
            hidden = F.silu(gate_up[:, : self.moe_hidden]) * gate_up[:, self.moe_hidden :]
            outputs.append(F.linear(hidden, down_w))
        if not outputs:
            return latent_sorted.new_zeros(0, _local(self.down_proj).shape[1])
        return torch.cat(outputs, dim=0)


# --------------------------------------------------------------------------------------------------- sublayers


class GDNMixer(nn.Module):
    """Gated DeltaNet mixer (four 128-wide heads, kernel-4 causal conv, swish-gated RMSNorm output)."""

    def __init__(self, args: Zonos3Args):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.dim, 4 * GDN_INNER + 2 * GDN_HEADS, bias=False)
        self.conv = _Conv(3 * GDN_INNER, GDN_KERNEL)
        self.out_weight = nn.Parameter(torch.ones(GDN_HEAD_DIM))
        # the release checkpoint stores both in bf16; the kernels read them as float32 (FSDP needs one dtype per layer)
        self.A_log = nn.Parameter(torch.zeros(GDN_HEADS))
        self.dt_bias = nn.Parameter(torch.zeros(GDN_HEADS))
        self.wo = nn.Linear(GDN_INNER, args.dim, bias=False)
        self._backend = None

    def reset_parameters(self, init_std=None, factor=1.0):
        del init_std, factor
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.conv.weight, std=0.02)
        nn.init.ones_(self.out_weight)
        nn.init.zeros_(self.A_log)
        nn.init.zeros_(self.dt_bias)
        nn.init.normal_(self.wo.weight, std=0.02)

    def backend(self, device: torch.device) -> str:
        if self._backend is None:
            wanted = self.args.gdn_backend
            if wanted == "auto":
                wanted = "fla" if device.type == "cuda" and fla_available() else "torch"
            if wanted == "fla" and not fla_available():
                raise RuntimeError("gdn_backend=fla needs the fla-core package (pip install fla-core==0.5.2)")
            self._backend = wanted
        return self._backend

    @torch.compiler.disable
    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        mixed = self.in_proj(x)
        qkv, z, a, b = mixed.split([3 * GDN_INNER, GDN_INNER, GDN_HEADS, GDN_HEADS], dim=-1)
        if self.backend(x.device) == "fla":
            return self._forward_fla(x, qkv, z, a, b, cu_seqlens)
        return self._forward_torch(x, qkv, z, a, b, cu_seqlens)

    def _forward_fla(self, x, qkv, z, a, b, cu_seqlens):
        from fla.modules.fused_norm_gate import rms_norm_gated
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        cu = cu_seqlens.to(torch.int32)
        qkv = self.conv.fla(qkv.unsqueeze(0).contiguous(), cu)
        q, k, v = [t.unflatten(-1, (GDN_HEADS, GDN_HEAD_DIM)) for t in qkv.split(GDN_INNER, dim=-1)]
        output, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=a.unsqueeze(0),
            beta=b.unsqueeze(0),
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu,
            state_v_first=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=_local(self.A_log).float(),
            dt_bias=_local(self.dt_bias).float(),
        )
        output = rms_norm_gated(
            output.reshape(-1, GDN_HEAD_DIM),
            z.reshape(-1, GDN_HEAD_DIM),
            _local(self.out_weight),
            None,
            activation="swish",
            eps=1e-5,
        )
        return self.wo(output.reshape(x.shape[0], GDN_INNER))

    def _forward_torch(self, x, qkv, z, a, b, cu_seqlens):
        """Portable reference: per-sequence causal depthwise conv and a float32 token-by-token recurrence."""
        conv = self.conv.torch(qkv, cu_seqlens)
        q, k, v = [part.reshape(len(x), GDN_HEADS, GDN_HEAD_DIM).float() for part in conv.split(GDN_INNER, -1)]
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        decay = torch.exp(-_local(self.A_log).float().exp() * F.softplus(a.float() + _local(self.dt_bias).float()))
        beta = b.float().sigmoid()
        outputs = []
        boundaries = cu_seqlens.tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            state = x.new_zeros((GDN_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM), dtype=torch.float32)
            for token in range(start, end):
                state = state * decay[token, :, None, None]
                prediction = (state * k[token, :, None, :]).sum(-1)
                delta = beta[token, :, None] * (v[token] - prediction)
                state = state + delta[:, :, None] * k[token, :, None, :]
                outputs.append((state * (q[token] * GDN_HEAD_DIM**-0.5)[:, None, :]).sum(-1))
        recurrent = torch.stack(outputs).to(x.dtype)
        normalized = F.rms_norm(recurrent, (GDN_HEAD_DIM,), weight=_local(self.out_weight), eps=1e-5)
        gated = (normalized.float() * F.silu(z.reshape(len(x), GDN_HEADS, GDN_HEAD_DIM).float())).to(x.dtype)
        return self.wo(gated.reshape(len(x), GDN_INNER))


class _Conv(nn.Module):
    """Depthwise causal convolution with SiLU; weight [channels, 1, kernel] as ``fla.modules.ShortConvolution``."""

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.channels, self.kernel = channels, kernel
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel))
        nn.init.normal_(self.weight, std=0.02)

    def fla(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        from fla.modules.conv import causal_conv1d

        out = causal_conv1d(
            x,
            _local(self.weight).squeeze(1),
            bias=None,
            activation="silu",
            cu_seqlens=cu_seqlens,
        )
        return out[0] if isinstance(out, tuple) else out

    def torch(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        weight = _local(self.weight)
        outputs = []
        boundaries = cu_seqlens.tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            sequence = x[start:end].T.unsqueeze(0)  # [1, C, T]
            padded = F.pad(sequence, (self.kernel - 1, 0))
            out = F.conv1d(padded, weight, groups=self.channels)
            outputs.append(out.squeeze(0).T)
        return F.silu(torch.cat(outputs, dim=0))


class Attention(nn.Module):
    """Gated global attention with QK RMSNorm, a learned per-head query temperature and interleaved RoPE."""

    def __init__(self, args: Zonos3Args):
        super().__init__()
        self.args = args
        self.n_heads, self.n_kv_heads, self.head_dim = args.n_heads, args.n_kv_heads, args.head_dim
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wkv = _Weight(2, args.n_kv_heads * args.head_dim, args.dim)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.temp = nn.Parameter(torch.ones(1, args.n_heads, 1))
        self.gater = nn.Linear(args.dim, args.n_heads, bias=False)

    def reset_parameters(self, init_std=None, factor=1.0):
        out_std = init_std or (0.02 / (2**0.5))
        trunc_normal_(self.wq.weight, std=0.02, a=-0.06, b=0.06)
        trunc_normal_(self.wkv.weight, std=0.02, a=-0.06, b=0.06)
        trunc_normal_(self.wo.weight, std=out_std / factor, a=-3 * out_std, b=3 * out_std)
        trunc_normal_(self.gater.weight, std=out_std / factor, a=-3 * out_std, b=3 * out_std)
        nn.init.ones_(self.temp)

    @torch.compiler.disable
    def _sdpa_per_sequence(self, q, k, v, cu_seqlens):
        if self.n_heads != self.n_kv_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        boundaries = cu_seqlens.tolist()
        outputs = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            q_seq = q[start:end].transpose(0, 1).unsqueeze(0)
            k_seq = k[start:end].transpose(0, 1).unsqueeze(0)
            v_seq = v[start:end].transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=True, scale=self.head_dim**-0.5)
            outputs.append(out.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        n, nh, nkv, hd = x.shape[0], self.n_heads, self.n_kv_heads, self.head_dim
        gate = torch.sigmoid(self.gater(x))
        q = self.wq(x).view(n, nh, hd)
        kv = self.wkv(x)
        k, v = kv[:, : nkv * hd].reshape(n, nkv, hd), kv[:, nkv * hd :].reshape(n, nkv, hd)
        q = F.rms_norm(q, (hd,), eps=1e-6) * _local(self.temp).abs().type_as(q)
        k = F.rms_norm(k, (hd,), eps=1e-6)
        q = rope_interleaved(q, positions, self.args.rope_theta)
        k = rope_interleaved(k, positions, self.args.rope_theta)
        out = self._sdpa_per_sequence(q, k, v, cu_seqlens)
        return self.wo((out * gate.unsqueeze(-1)).reshape(n, nh * hd))


class DenseFFN(nn.Module):
    def __init__(self, args: Zonos3Args):
        super().__init__()
        self.hidden = args.ffn_hidden
        self.w_in = _Weight(2, args.ffn_hidden, args.dim)
        self.w_out = nn.Linear(args.ffn_hidden, args.dim, bias=False)

    def reset_parameters(self, init_std=0.02, factor=1.0):
        trunc_normal_(self.w_in.weight, std=0.02, a=-0.06, b=0.06)
        trunc_normal_(self.w_out.weight, std=init_std, a=-3 * init_std, b=3 * init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.w_in(x)
        return self.w_out(flat[:, : self.hidden] * F.silu(flat[:, self.hidden :]))


class Router(nn.Module):
    """Linear router over the RMS-normalized input; expert choice = top-k of (logits - balancing bias)."""

    def __init__(self, args: Zonos3Args):
        super().__init__()
        self.args = args
        self.router_linear = nn.Linear(args.dim, args.n_experts, bias=False)
        self.rmsnorm_eda = nn.RMSNorm(args.dim, eps=args.norm_eps)
        # Quantile balancing is a state update, not a learned parameter: a float32 buffer (the checkpoint key
        # ``router.balancing_biases``) that only ``Zonos3LM.quantile_balancing_step`` changes; FSDP's bf16 parameter
        # cast never touches it.
        self.register_buffer("balancing_biases", torch.zeros(args.n_experts, dtype=torch.float32))
        self.register_buffer("qb_accum", torch.zeros(args.n_experts, dtype=torch.float32), persistent=False)
        self.register_buffer("qb_count", torch.zeros((), dtype=torch.long), persistent=False)
        self.collect_qb = False

    def reset_parameters(self, init_std=0.02):
        self.router_linear.reset_parameters()
        self.rmsnorm_eda.reset_parameters()
        nn.init.zeros_(self.balancing_biases)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router_linear(self.rmsnorm_eda(x)).float()
        bias = _local(self.balancing_biases).detach().float()
        top_k = self.args.moe_topk
        collect = self.collect_qb and self.training and torch.is_grad_enabled()
        with torch.no_grad():
            if collect:
                # Exact QB coordinate update (the trainer's quantile balancing): the order statistic that targets
                # tokens * top_k / experts routes per expert, applied after the optimizer step.
                scores = logits.detach()
                topk = torch.topk(scores - bias, top_k + 1, dim=-1)
                choices = topk.indices[:, :top_k]
                tokens, experts = scores.shape
                if (tokens * top_k) % experts == 0:
                    target = tokens * top_k // experts
                    alpha = topk.values[:, -1:]
                    beta_local = torch.topk(scores - alpha, target + 1, dim=0).values[-1]
                    self.qb_accum += beta_local
                    self.qb_count += 1
            else:
                choices = torch.topk(logits.detach() - bias, top_k, dim=-1).indices
        weights = torch.softmax(logits.gather(-1, choices), dim=-1)
        return choices, weights


class MoEFFN(nn.Module):
    def __init__(self, args: Zonos3Args):
        super().__init__()
        self.args = args
        self.experts = _Experts(args.n_experts, args.moe_hidden, args.moe_routed_dim)
        self.latent_down = nn.Linear(args.dim, args.moe_routed_dim, bias=False)
        self.latent_up = nn.Linear(args.moe_routed_dim, args.dim, bias=False)
        self.router = Router(args)

    def reset_parameters(self, init_std=0.02):
        trunc_normal_(self.experts.gate_up_proj, std=0.02, a=-0.06, b=0.06)
        trunc_normal_(self.experts.down_proj, std=init_std, a=-3 * init_std, b=3 * init_std)
        trunc_normal_(self.latent_down.weight, std=0.02, a=-0.06, b=0.06)
        trunc_normal_(self.latent_up.weight, std=init_std, a=-3 * init_std, b=3 * init_std)
        self.router.reset_parameters(init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        choices, weights = self.router(x)  # [T, K]
        top_k = self.args.moe_topk
        latent = self.latent_down(x)  # [T, routed]
        flat_choices = choices.reshape(-1)
        order = torch.argsort(flat_choices, stable=True)
        tokens_per_expert = torch.bincount(flat_choices, minlength=self.args.n_experts)
        routed_input = latent[order // top_k]
        routed_output = self.experts(routed_input, tokens_per_expert)
        out = routed_output.new_zeros(latent.shape[0] * top_k, latent.shape[1])
        out[order] = routed_output
        out = out.view(latent.shape[0], top_k, latent.shape[1])
        combined = (out.float() * weights.unsqueeze(-1)).sum(dim=1).to(x.dtype)
        return self.latent_up(combined)


class TransformerBlock(nn.Module):
    def __init__(self, args: Zonos3Args, layer: int):
        super().__init__()
        self.layer = layer
        self.weight_init_std = args.init_base_std / (2 * args.n_layers) ** 0.5
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.attention = GDNMixer(args) if args.uses_gdn(layer) else Attention(args)
        self.feed_forward = DenseFFN(args) if layer == 0 else MoEFFN(args)

    def init_weights(self):
        self.attention.reset_parameters(self.weight_init_std)
        self.feed_forward.reset_parameters(self.weight_init_std)
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        sources: list,
        normalized: list,
        query_pair: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The layer's half of Full AttnRes: (attention/GDN output, FFN/MoE output), each read from a softmax mix of
        every earlier source. Called as a module so FSDP's hooks materialize the layer's weights."""
        dtype = sources[0].dtype
        mixed = full_attnres_route(sources, normalized, query_pair[0], dtype)
        attn_in = self.attention_norm(mixed)
        if isinstance(self.attention, GDNMixer):
            attention = self.attention(attn_in, cu_seqlens)
        else:
            attention = self.attention(attn_in, cu_seqlens, positions)
        attention_normalized = F.rms_norm(attention, (attention.shape[-1],), eps=eps)
        mixed = full_attnres_route(sources + [attention], normalized + [attention_normalized], query_pair[1], dtype)
        ffn_out = self.feed_forward(self.ffn_norm(mixed))
        return attention, ffn_out


class MultiEmbedding(nn.Module):
    def __init__(self, args: Zonos3Args):
        super().__init__()
        self.n_codebooks = args.n_codebooks
        self.embedders = nn.ModuleList(
            [nn.Embedding(args.audio_vocab, args.dim, padding_idx=args.audio_pad_id) for _ in range(args.n_codebooks)]
        )
        self.embedders.append(nn.Embedding(args.text_rows, args.dim, padding_idx=args.text_vocab))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return sum(self.embedders[i](codes[..., i]) for i in range(codes.size(-1)))

    def init_weights(self):
        for embedder in self.embedders:
            embedder.reset_parameters()


# --------------------------------------------------------------------------------------------------- model


def full_attnres_route(sources: list, normalized: list, query: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Softmax mix of the bank: scores = RMS-normalized source . query (float32), weights over the sources."""
    scores = torch.stack([(n * query).sum(-1, dtype=torch.float32) for n in normalized])  # [S, T]
    weights = scores.softmax(0).to(dtype)
    mixed = sources[0] * weights[0].unsqueeze(-1)
    for source, weight in zip(sources[1:], weights[1:]):
        mixed = mixed + source * weight.unsqueeze(-1)
    return mixed


class Zonos3LM(nn.Module):
    def __init__(self, args: Zonos3Args):
        super().__init__()
        self.args = args
        self.n_codebooks, self.audio_vocab = args.n_codebooks, args.audio_vocab
        self.multi_embedder = MultiEmbedding(args)
        self.layers = nn.ModuleList(TransformerBlock(args, layer) for layer in range(args.n_layers))
        self.out_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.multi_output = nn.Linear(args.dim, args.n_codebooks * args.audio_vocab, bias=False)
        self.block_attnres_queries = nn.Parameter(torch.zeros(args.n_sources, args.dim))
        self.speaker_projection = nn.Linear(args.speaker_embedding_dim, args.dim, bias=True)
        self.condition_embeddings = nn.ModuleList(nn.Embedding(n + 1, args.dim) for n in args.condition_bucket_counts)
        self.soft_cap = partial(softcap, cap=args.loss_softcap) if args.loss_softcap > 0 else nn.Identity()
        self.reset_parameters()

    # -- init ---------------------------------------------------------------------------------------------------------

    def reset_parameters(self):
        init_std = self.args.init_base_std
        for layer in self.layers:
            layer.init_weights()
        self.out_norm.reset_parameters()
        self.multi_embedder.init_weights()
        trunc_normal_(self.multi_output.weight, std=init_std, a=-3 * init_std, b=3 * init_std)
        nn.init.zeros_(self.block_attnres_queries)
        trunc_normal_(self.speaker_projection.weight, std=0.02, a=-0.06, b=0.06)
        nn.init.zeros_(self.speaker_projection.bias)
        for table in self.condition_embeddings:
            table.reset_parameters()

    # -- quantile balancing -------------------------------------------------------------------------------------------

    def routers(self) -> list[Router]:
        return [layer.feed_forward.router for layer in self.layers if isinstance(layer.feed_forward, MoEFFN)]

    def set_quantile_balancing(self, enabled: bool) -> None:
        for router in self.routers():
            router.collect_qb = enabled

    @torch.no_grad()
    def quantile_balancing_step(self, ema: float = 0.0, process_group=None) -> None:
        """Apply the accumulated QB estimates (one synchronized update per optimizer step, centered per layer)."""
        import torch.distributed as dist

        routers = self.routers()
        if not routers:
            return
        accum = torch.stack([r.qb_accum for r in routers])
        counts = torch.stack([r.qb_count for r in routers]).to(accum.dtype).unsqueeze(-1)
        packed = torch.cat([accum, counts], dim=-1)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=process_group)
        if float(packed[:, -1].sum()) == 0:
            return
        estimate = packed[:, :-1] / packed[:, -1:].clamp_min(1.0)
        current = torch.stack([_local(r.balancing_biases).float() for r in routers])
        new = ema * current + (1.0 - ema) * estimate
        new = new - new.mean(dim=-1, keepdim=True)
        for router, beta in zip(routers, new):
            _local(router.balancing_biases).copy_(beta)
            router.qb_accum.zero_()
            router.qb_count.zero_()

    # -- forward ------------------------------------------------------------------------------------------------------

    def embed(
        self,
        token_values: torch.Tensor,
        speaker_emb_values: Optional[torch.Tensor],
        speaker_token_positions: Optional[torch.Tensor],
        condition_values: Optional[torch.Tensor],
        condition_token_positions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.multi_embedder(token_values)
        if speaker_emb_values is not None and speaker_emb_values.shape[0]:
            projected = self.speaker_projection(speaker_emb_values.to(self.speaker_projection.weight.dtype))
            h = h.index_copy(0, speaker_token_positions, projected.to(h.dtype))
        if condition_values is not None and condition_values.shape[0]:
            for index, table in enumerate(self.condition_embeddings):
                h = h.index_copy(0, condition_token_positions[:, index], table(condition_values[:, index]).to(h.dtype))
        return F.rms_norm(h, (h.shape[-1],), eps=self.args.norm_eps)

    def forward(
        self,
        token_values: torch.Tensor,  # [tokens, n_codebooks + 1], the last column is text
        target: Optional[torch.Tensor] = None,  # [tokens, n_codebooks], -100 = unsupervised
        cu_seqlens: Optional[torch.Tensor] = None,  # int32 [sequences + 1]
        maxlen: Optional[int] = None,
        speaker_emb_values: Optional[torch.Tensor] = None,  # float32 [n, speaker_embedding_dim]
        speaker_token_positions: Optional[torch.Tensor] = None,  # long [n]
        condition_values: Optional[torch.Tensor] = None,  # long [m, condition_count]
        condition_token_positions: Optional[torch.Tensor] = None,  # long [m, condition_count]
    ):
        del maxlen
        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, token_values.shape[0]], dtype=torch.int32, device=token_values.device)
        cu_seqlens = cu_seqlens.to(torch.int32)
        positions = sequence_positions(cu_seqlens)
        x = self.embed(
            token_values, speaker_emb_values, speaker_token_positions, condition_values, condition_token_positions
        )
        eps, dtype = self.args.norm_eps, x.dtype
        queries = _local(self.block_attnres_queries)
        sources = [x]
        normalized = [F.rms_norm(x, (x.shape[-1],), eps=eps)]
        for layer, block in enumerate(self.layers):
            attention, ffn_out = block(sources, normalized, queries[2 * layer : 2 * layer + 2], cu_seqlens, positions, eps)
            sources += [attention, ffn_out]
            normalized += [
                F.rms_norm(attention, (attention.shape[-1],), eps=eps),
                F.rms_norm(ffn_out, (ffn_out.shape[-1],), eps=eps),
            ]
        mixed = full_attnres_route(sources, normalized, queries[-1], dtype)
        logits = self.multi_output(self.out_norm(mixed)).unflatten(-1, (self.n_codebooks, self.audio_vocab))
        logits = self.soft_cap(logits)
        loss = None if target is None else self.heterogeneous_audio_loss(logits, target)
        return logits, loss, logits.new_zeros((), dtype=torch.float32)

    def heterogeneous_audio_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross entropy over each codebook's ordinary codes plus EOA (never padding), codebook 0 weighted."""
        size, eoa = self.args.codebook_size, self.args.eoa_id
        loss_sum = logits.new_zeros((), dtype=torch.float32)
        weight_sum = logits.new_zeros((), dtype=torch.float32)
        for index in range(self.n_codebooks):
            codebook_weight = self.args.semantic_codebook_loss_weight if index == 0 else 1.0
            codebook_logits = logits[:, index].float()
            ordinary = codebook_logits[:, :size]
            eoa_logits = codebook_logits[:, eoa]
            log_normalizer = torch.logaddexp(torch.logsumexp(ordinary, dim=-1), eoa_logits)
            codebook_target = target[:, index]
            supervised = codebook_target != -100
            ordinary_index = codebook_target.clamp(min=0, max=size - 1)
            target_logits = ordinary.gather(1, ordinary_index.unsqueeze(1)).squeeze(1)
            target_logits = torch.where(codebook_target == eoa, eoa_logits, target_logits)
            loss_sum = loss_sum + codebook_weight * torch.where(supervised, log_normalizer - target_logits, 0.0).sum()
            weight_sum = weight_sum + codebook_weight * supervised.sum()
        return loss_sum / weight_sum.clamp_min(1.0)


# --------------------------------------------------------------------------------------------------- utilities


def fla_available() -> bool:
    try:
        import fla  # noqa: F401
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
    except Exception:
        return False
    return True


def build_fsdp_grouping_plan(args: Zonos3Args):
    return [(f"layers.{i}", True) for i in range(args.n_layers)] + [("multi_output", False)]
