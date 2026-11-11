"""Small LoRA implementation for the ZONOS3 trainer (attention / GDN projections, FFN and experts, output head)."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from model import _Experts, _Weight

TARGET_PRESETS = ("attention", "attention-output", "all")


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    targets: str = "attention-output"
    layer_start: int | None = None
    layer_end: int | None = None

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def to_dict(self) -> dict:
        return asdict(self) | {"scale": self.scale}


class LoRALinear(nn.Linear):
    """An identity-initialized LoRA branch with release-compatible base keys."""

    def __init__(self, in_features, out_features, bias, rank, alpha, dropout, *, device, dtype):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.lora_a = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @classmethod
    def from_linear(cls, module: nn.Linear, config: LoraConfig) -> "LoRALinear":
        replacement = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            config.rank,
            config.alpha,
            config.dropout,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        replacement.weight = module.weight
        replacement.bias = module.bias
        replacement.weight.requires_grad_(False)
        if replacement.bias is not None:
            replacement.bias.requires_grad_(False)
        return replacement

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        adapter = F.linear(F.linear(self.lora_dropout(inputs), self.lora_a), self.lora_b)
        return base + adapter * self.lora_scale


_ATTENTION = (".attention.wq", ".attention.wkv", ".attention.wo", ".attention.in_proj")
_FFN = (".feed_forward.w_in", ".feed_forward.w_out", ".feed_forward.latent_down", ".feed_forward.latent_up")


def _is_target(name: str, preset: str) -> bool:
    if preset == "attention":
        return name.endswith(_ATTENTION)
    if preset == "attention-output":
        return name.endswith(_ATTENTION) or name == "multi_output"
    if preset == "all":
        return name.endswith(_ATTENTION + _FFN) or name.endswith(".feed_forward.experts") or name == "multi_output"
    raise ValueError(f"Unknown LoRA target preset: {preset}")


def _layer_index(name: str) -> int | None:
    match = re.match(r"layers\.(\d+)\.", name)
    return None if match is None else int(match.group(1))


def _is_in_layer_range(name: str, config: LoraConfig) -> bool:
    layer = _layer_index(name)
    if layer is None:
        return True
    if config.layer_start is not None and layer < config.layer_start:
        return False
    return config.layer_end is None or layer < config.layer_end


def _replace_submodule(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if child_name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def apply_lora(model: nn.Module, config: LoraConfig) -> list[str]:
    """Freeze the base and inject adapters selected by a named preset."""
    if config.rank < 1:
        raise ValueError("LoRA rank must be positive")
    if config.alpha <= 0:
        raise ValueError("LoRA alpha must be positive")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1)")
    if config.layer_start is not None and config.layer_start < 0:
        raise ValueError("LoRA layer start must be non-negative")
    if config.layer_end is not None and config.layer_end <= 0:
        raise ValueError("LoRA layer end must be positive")
    if config.layer_start is not None and config.layer_end is not None and config.layer_start >= config.layer_end:
        raise ValueError("LoRA layer range must be non-empty")
    if config.targets not in TARGET_PRESETS:
        raise ValueError(f"Unknown LoRA target preset: {config.targets}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    targeted = []
    for name, module in list(model.named_modules()):
        if not _is_target(name, config.targets) or not _is_in_layer_range(name, config):
            continue
        if isinstance(module, LoRALinear):
            raise RuntimeError(f"LoRA is already enabled for {name}")
        if isinstance(module, nn.Linear):
            _replace_submodule(model, name, LoRALinear.from_linear(module, config))
        elif isinstance(module, _Weight):
            module.enable_lora(config.rank, config.alpha, config.dropout)
        elif isinstance(module, _Experts):
            module.enable_lora(config.rank, config.alpha)
        else:
            continue
        targeted.append(name)
    if not targeted:
        raise RuntimeError(f"LoRA preset {config.targets!r} selected no modules")
    return targeted


def is_adapter_key(name: str) -> bool:
    return name.endswith(("lora_a", "lora_b"))


def adapter_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in state.items() if is_adapter_key(name)}


def _base_key(a_key: str) -> str:
    """The base weight of an adapter A tensor: ``x.lora_a`` -> ``x.weight``; ``experts.gate_up_proj_lora_a`` ->
    ``experts.gate_up_proj``."""
    if a_key.endswith("_lora_a"):
        return a_key[: -len("_lora_a")]
    return a_key[: -len(".lora_a")] + ".weight"


def merge_lora_state_dict(state: dict[str, torch.Tensor], config: LoraConfig) -> dict[str, torch.Tensor]:
    """Merge adapters into a release-layout state dict and drop the adapter keys."""
    merged = {name: value for name, value in state.items() if not is_adapter_key(name)}
    a_keys = sorted(name for name in state if name.endswith("lora_a"))
    b_keys = sorted(name for name in state if name.endswith("lora_b"))
    if len(a_keys) != len(b_keys):
        raise RuntimeError(f"Unpaired LoRA tensors: {len(a_keys)} A tensors, {len(b_keys)} B tensors")
    for a_key in a_keys:
        b_key = a_key[: -len("lora_a")] + "lora_b"
        weight_key = _base_key(a_key)
        if b_key not in state or weight_key not in merged:
            raise RuntimeError(f"Incomplete LoRA state for {a_key}")
        weight = merged[weight_key]
        a, b = state[a_key].float(), state[b_key].float()
        if a.ndim == 3:  # per-expert [E, r, in] x [E, out, r]
            update = torch.bmm(b, a)
        else:
            update = torch.matmul(b, a)
        update = update.reshape(weight.shape).mul_(config.scale).to(weight.dtype)
        merged[weight_key] = weight + update
    return merged
