"""Gauge-aware LoRA-Muon optimizer.

This implements Algorithm 1 from Cesista et al., "LoRA-Muon: Spectral
Steepest Descent on the Low-Rank Manifold" (arXiv:2606.12921). The repository's
LoRA convention is delta_W = scale * B @ A, with A shaped (rank, input) and B
shaped (output, rank); the paper writes the same product as U @ V.T.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor, Replicate

POLAR_EXPRESS_COEFFICIENTS = (
    (7.2086, -15.5131, 9.0178),
    (3.9623, -2.5813, 0.4542),
    (3.9466, -2.5765, 0.4544),
    (3.8991, -2.5671, 0.4566),
    (3.7186, -2.5308, 0.4653),
    (3.1390, -2.3073, 0.4733),
    (2.1715, -1.5246, 0.3885),
    (1.8648, -1.2224, 0.3577),
)


def matrix_sign(matrix: torch.Tensor) -> torch.Tensor:
    """Polar-Express approximation to the rectangular matrix-sign map."""
    if not torch.count_nonzero(matrix):
        return torch.zeros_like(matrix)
    transposed = matrix.shape[0] > matrix.shape[1]
    value = matrix.T if transposed else matrix
    value = value.bfloat16()
    value = value / value.norm().clamp_min(1e-20)
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
        gram = value @ value.T
        value = a * value + (b * gram + c * (gram @ gram)) @ value
    value = value.T if transposed else value
    return value.float()


def matrix_invroot(matrix: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Stable inverse square root for the small rank-by-rank Gram matrices."""
    value = matrix.float()
    scale = value.norm().clamp_min(1.0)
    normalized = value / scale
    identity = torch.eye(value.shape[0], device=value.device, dtype=value.dtype)
    eigenvalues, eigenvectors = torch.linalg.eigh(normalized + epsilon * identity)
    inverse = (eigenvectors * eigenvalues.clamp_min(epsilon).rsqrt().unsqueeze(0)) @ eigenvectors.T
    return inverse / scale.sqrt()


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


@torch.no_grad()
def _apply_full_delta(parameter: torch.Tensor, delta: torch.Tensor) -> None:
    delta = delta.to(parameter.dtype)
    if isinstance(parameter, DTensor):
        replicated = DTensor.from_local(
            delta,
            device_mesh=parameter.device_mesh,
            placements=[Replicate()],
            run_check=False,
            shape=parameter.shape,
            stride=parameter.stride(),
        )
        parameter.add_(replicated.redistribute(placements=parameter.placements))
    else:
        parameter.add_(delta)


@dataclass
class _PairState:
    name: str
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    momentum_a: torch.Tensor | None = None
    momentum_b: torch.Tensor | None = None


class LoraMuon:
    """Low-rank-manifold Muon for paired LoRA factors."""

    def __init__(
        self,
        pairs: list[tuple[str, torch.Tensor, torch.Tensor]],
        *,
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
    ) -> None:
        if not pairs:
            raise ValueError("LoRA-Muon requires at least one adapter pair")
        self.pairs = [_PairState(name, lora_a, lora_b) for name, lora_a, lora_b in pairs]
        self.momentum = momentum
        self.param_groups = [
            {
                "params": [parameter for pair in self.pairs for parameter in (pair.lora_a, pair.lora_b)],
                "lr": lr,
                "weight_decay": weight_decay,
                "algorithm": "lora-muon-manifold",
            }
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.param_groups[0]["params"]:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        learning_rate = float(self.param_groups[0]["lr"])
        weight_decay = float(self.param_groups[0]["weight_decay"])
        if weight_decay and learning_rate * weight_decay >= 1:
            raise ValueError("LoRA-Muon requires learning_rate * weight_decay < 1")
        decay_scale = math.sqrt(1.0 - learning_rate * weight_decay)

        for pair in self.pairs:
            if pair.lora_a.grad is None or pair.lora_b.grad is None:
                continue
            lora_a = _full(pair.lora_a).float()
            lora_b = _full(pair.lora_b).float()
            gradient_a = _full(pair.lora_a.grad).float()
            gradient_b = _full(pair.lora_b.grad).float()
            if pair.momentum_a is None:
                pair.momentum_a = torch.zeros_like(gradient_a)
                pair.momentum_b = torch.zeros_like(gradient_b)
            pair.momentum_a.lerp_(gradient_a, 1.0 - self.momentum)
            pair.momentum_b.lerp_(gradient_b, 1.0 - self.momentum)

            # Paper convention U @ V.T maps to this repository's B @ A:
            # U=B, V=A.T, S_U=B.T@B, S_V=A@A.T.
            inverse_u = matrix_invroot(lora_b.T @ lora_b)
            inverse_v = matrix_invroot(lora_a @ lora_a.T)
            delta_b = -0.5 * learning_rate * (matrix_sign(pair.momentum_b @ inverse_v) @ inverse_v)
            delta_v = -0.5 * learning_rate * (matrix_sign(pair.momentum_a.T @ inverse_u) @ inverse_u)
            delta_a = delta_v.T
            _apply_full_delta(
                pair.lora_a,
                (decay_scale - 1.0) * lora_a + delta_a / decay_scale,
            )
            _apply_full_delta(
                pair.lora_b,
                (decay_scale - 1.0) * lora_b + delta_b / decay_scale,
            )
