#!/usr/bin/env python3
"""Minimal FSDP2 fine-tuning for the public ZONOS3-Mini checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tomllib
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader, DistributedSampler, Subset

from data import DrawPolicy, EncodedDataset, PackingCollator, TokenConfig
from lora import LoraConfig, adapter_state_dict, apply_lora, merge_lora_state_dict
from model import Zonos3Args, Zonos3LM, _Experts, _Weight

BASE_MODEL_REPO = "Zyphra/ZONOS3-Mini"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description="Fine-tune ZONOS3-Mini with a TOML recipe and optional CLI overrides.")
    parser.add_argument("--config", type=Path, help="TOML training recipe")
    parser.add_argument("--base-model", type=Path, default=Path("artifacts/base_model"))
    parser.add_argument("--data", type=Path, nargs="+", default=[Path("data/processed/train.pt")], help="Processed .pt file(s)")
    parser.add_argument("--output", type=Path, default=Path("outputs/run/checkpoint"))
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4, help="Examples per GPU per micro-batch")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=8192, help="Longest packed example (rows)")
    parser.add_argument("--lr", type=float, default=3.5e-4, help="AdamW learning rate of the release recipe")
    parser.add_argument("--muon-lr", type=float, default=3.5e-3, help="Muon learning rate of the release recipe")
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=0.05,
        help="Fine-tuning multiplier applied to every LR from the release training config.",
    )
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=333)
    parser.add_argument("--split-seed", type=int, help="Dataset-split seed; defaults to --seed.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--finetune-method", choices=("full", "lora"), default="full")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-layer-start", type=int, help="Optional first transformer layer to adapt (inclusive).")
    parser.add_argument("--lora-layer-end", type=int, help="Optional final transformer layer boundary (exclusive).")
    parser.add_argument("--lora-targets", choices=("attention", "attention-output", "all"), default="attention-output")
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--lora-plus-ratio", type=float, default=16.0, help="LoRA B-to-A LR ratio; 1 disables LoRA+.")
    parser.add_argument("--lora-weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-optimizer", choices=("adamw", "lora-muon-manifold"), default="adamw")
    # the conditioning mix of each draw (data.DrawPolicy)
    parser.add_argument("--p-embedding", type=float, default=0.9, help="Share of draws with the speaker-embedding row")
    parser.add_argument("--p-reference", type=float, default=0.0, help="Share of draws with a raw reference prefix of another clip")
    parser.add_argument("--embedding-other-clip", type=float, default=0.0, help="Share of speaker rows that carry another clip's embedding")
    parser.add_argument("--condition-dropout", type=float, default=0.2)
    parser.add_argument("--reference-min-seconds", type=float, default=1.0)
    parser.add_argument("--reference-max-seconds", type=float, default=30.0)
    parser.add_argument("--semantic-loss-weight", type=float, default=3.0, help="Loss weight of codebook 0")
    parser.add_argument(
        "--qb-update",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Quantile-balancing update of the MoE router biases after every step (default: off; see README).",
    )
    parser.add_argument("--qb-ema", type=float, default=0.99, help="EMA of the router-bias update when --qb-update is on")
    parser.add_argument("--gdn-backend", choices=("auto", "fla", "torch"), default="auto")
    parser.add_argument("--val-fraction", type=float, default=None, help="Defaults to 0.1 for LoRA and 0 for full.")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=0, help="Maximum validation batches per pass; 0 = all.")
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Every N steps: adapter snapshots (LoRA) or a resumable training state (full fine-tunes); 0 disables.",
    )
    parser.add_argument("--keep-checkpoints", type=int, default=2, help="Resumable training states to keep")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a full fine-tune from the latest training state under the output's run directory.",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the main forward around the kernel boundaries (GDN, experts, attention loops stay eager).",
    )
    parser.add_argument(
        "--export-merged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the public-inference model.pth. Sweeps can disable this and retain only adapters.",
    )
    if config_args.config is not None:
        with config_args.config.open("rb") as stream:
            defaults = tomllib.load(stream)
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(defaults) - valid_keys)
        if unknown_keys:
            parser.error(f"{config_args.config} contains unknown options: {', '.join(unknown_keys)}")
        for key in ("base_model", "output"):
            if key in defaults:
                defaults[key] = Path(defaults[key])
        if "data" in defaults:
            value = defaults["data"]
            defaults["data"] = [Path(v) for v in (value if isinstance(value, list) else [value])]
        parser.set_defaults(**defaults)

    args = parser.parse_args(argv)
    if args.val_fraction is None:
        args.val_fraction = 0.1 if args.finetune_method == "lora" else 0.0
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.qb_update is None:
        args.qb_update = False  # the release biases stay; per-step rebalancing on small batches drifts them (README)
    if not 0.0 <= args.val_fraction < 1.0:
        parser.error("--val-fraction must be in [0, 1)")
    if args.p_embedding + args.p_reference > 1.0 + 1e-9:
        parser.error("--p-embedding + --p-reference must not exceed 1")
    return args


def load_params(base_model: Path) -> dict:
    params = json.loads((base_model / "params.json").read_text(encoding="utf-8"))
    return params


def release_model_args(params: dict, args: argparse.Namespace) -> Zonos3Args:
    model_args = Zonos3Args.from_params(params)
    model_args.max_seqlen = min(model_args.max_seqlen, args.max_seq_len)
    model_args.semantic_codebook_loss_weight = args.semantic_loss_weight
    model_args.gdn_backend = args.gdn_backend
    return model_args


def apply_fsdp(model: Zonos3LM, mesh):
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    return fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)


def release_to_training_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The trainer uses the release layout as is (every tensor name and shape of model.pth)."""
    return {key: (value if value.is_contiguous() else value.contiguous()) for key, value in state.items()}


def training_to_release_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return dict(state)


def load_release_weights(model: Zonos3LM, checkpoint: Path, rank: int, *, allow_missing_lora: bool = False) -> None:
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True) if rank == 0 else {}
    if rank == 0:
        state = release_to_training_state(state)
    incompatible = set_model_state_dict(
        model,
        state,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=not allow_missing_lora),
    )
    missing = [
        name
        for name in incompatible.missing_keys
        if not allow_missing_lora or not name.endswith(("lora_a", "lora_b"))
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(f"Incompatible release checkpoint: {incompatible}")
    del state


def _muon_parameter_ids(model: nn.Module) -> set[int]:
    ids = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            continue
        if isinstance(module, nn.Linear) and module.weight.requires_grad:
            ids.add(id(module.weight))
        elif isinstance(module, _Weight) and module.weight.requires_grad:
            ids.add(id(module.weight))
        elif isinstance(module, _Experts):
            for parameter in (module.gate_up_proj, module.down_proj):
                if parameter.requires_grad:
                    ids.add(id(parameter))
    return ids


def _use_muon(name: str, parameter: torch.Tensor, muon_parameter_ids: set[int]) -> bool:
    secretly_1d = sum(dimension > 1 for dimension in parameter.shape) <= 1
    return (
        id(parameter) in muon_parameter_ids
        and not secretly_1d
        and parameter.ndim >= 2
        and "multi_output" not in name
        and ".gater." not in name
    )


def _optimizer_groups(model: nn.Module, args: argparse.Namespace) -> list[dict]:
    muon_parameter_ids = _muon_parameter_ids(model)
    grouped: dict[tuple[str, float, float, bool], list[torch.Tensor]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _use_muon(name, parameter, muon_parameter_ids):
            key = ("muon", args.muon_lr * args.lr_scale, args.weight_decay, True)
        else:
            key = ("adamw", args.lr * args.lr_scale, args.weight_decay, True)
        grouped[key].append(parameter)
    groups = []
    for (algorithm, lr, weight_decay, scheduled), parameters in sorted(grouped.items(), key=lambda item: item[0]):
        groups.append(
            {
                "params": parameters,
                "algorithm": algorithm,
                "lr": lr,
                "base_lr": lr,
                "weight_decay": weight_decay,
                "scheduled": scheduled,
            }
        )
    return groups


class OffloadedMuon(torch.optim.Optimizer):
    """Single-GPU Muon with BF16 momentum on CPU and matrix-wise GPU updates."""

    def __init__(self, groups: list[dict], momentum: float = 0.95, epsilon: float = 1e-8):
        super().__init__(groups, {"momentum": momentum, "epsilon": epsilon})
        for group in self.param_groups:
            group.setdefault("momentum", momentum)
            group.setdefault("epsilon", epsilon)
            for parameter in group["params"]:
                local_parameter = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                self.state[parameter]["momentum"] = torch.zeros_like(local_parameter, device="cpu")

    @torch.no_grad()
    def step(self, closure=None):
        from dion.megabatch_base import adjust_lr_spectral_norm
        from dion.polar_express import polar_express_triton

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum_coefficient = float(group["momentum"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                local_parameter = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                local_gradient = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
                cpu_momentum = self.state[parameter]["momentum"]
                parameter_matrices = local_parameter.reshape(-1, *local_parameter.shape[-2:])
                gradient_matrices = local_gradient.reshape(-1, *local_gradient.shape[-2:])
                momentum_matrices = cpu_momentum.reshape(-1, *cpu_momentum.shape[-2:])
                adjusted_lr = adjust_lr_spectral_norm(lr, local_parameter.shape, flatten=False)
                for parameter_matrix, gradient_matrix, cpu_momentum_matrix in zip(
                    parameter_matrices, gradient_matrices, momentum_matrices
                ):
                    gpu_momentum = cpu_momentum_matrix.to(device=gradient_matrix.device, non_blocking=False)
                    gpu_momentum.mul_(momentum_coefficient).add_(gradient_matrix.to(dtype=gpu_momentum.dtype))
                    cpu_momentum_matrix.copy_(gpu_momentum, non_blocking=False)
                    update = polar_express_triton(gpu_momentum, epsilon=float(group["epsilon"]))
                    parameter_matrix.mul_(1.0 - lr * weight_decay)
                    parameter_matrix.add_(update, alpha=-adjusted_lr)
        return loss


class ChainedOptimizer:
    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers
        self.param_groups = [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()


def build_optimizer(model: nn.Module, args: argparse.Namespace, mesh):
    if args.finetune_method == "lora":
        named_lora = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.endswith(("lora_a", "lora_b"))
        ]
        lora_a = [parameter for name, parameter in named_lora if name.endswith("lora_a")]
        lora_b = [parameter for name, parameter in named_lora if name.endswith("lora_b")]
        if not lora_a or not lora_b:
            raise RuntimeError("LoRA optimizer found no paired adapter parameters")
        if args.lora_optimizer == "adamw":
            if args.lora_plus_ratio <= 0:
                raise ValueError("--lora-plus-ratio must be positive")
            groups = [
                {"params": lora_a, "lr": args.lora_lr},
                {"params": lora_b, "lr": args.lora_lr * args.lora_plus_ratio},
            ]
            optimizer = torch.optim.AdamW(
                groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.lora_weight_decay, fused=True
            )
        else:
            from lora_muon import LoraMuon

            if args.lora_targets != "attention":
                raise ValueError("Manifold LoRA-Muon supports --lora-targets attention; use AdamW for the others")
            by_prefix: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
            for name, parameter in named_lora:
                factor = "a" if name.endswith("lora_a") else "b"
                by_prefix[name[: -len("lora_a")]][factor] = parameter
            incomplete = [prefix for prefix, pair in by_prefix.items() if set(pair) != {"a", "b"}]
            if incomplete:
                raise RuntimeError(f"Unpaired LoRA-Muon adapters: {incomplete[:5]}")
            pairs = [(prefix, pair["a"], pair["b"]) for prefix, pair in sorted(by_prefix.items())]
            optimizer = LoraMuon(pairs, lr=args.lora_lr, momentum=0.95, weight_decay=args.lora_weight_decay)
        for group in optimizer.param_groups:
            group["base_lr"] = group["lr"]
            group["scheduled"] = True
        if dist.get_rank() == 0:
            trainable = sum(parameter.numel() for parameter in (*lora_a, *lora_b))
            lr_detail = (
                f"manifold_lr={args.lora_lr:.6g}"
                if args.lora_optimizer == "lora-muon-manifold"
                else f"lora_a_lr={args.lora_lr:.6g} lora_b_lr={args.lora_lr * args.lora_plus_ratio:.6g}"
            )
            print(
                f"optimizer={args.lora_optimizer} trainable_parameters={trainable:,} {lr_detail} "
                f"wd={args.lora_weight_decay}",
                flush=True,
            )
        return optimizer

    from dion import Muon

    groups = _optimizer_groups(model, args)
    optimizer_kwargs = {
        "distributed_mesh": mesh,
        "lr": args.muon_lr * args.lr_scale,
        "betas": (0.9, 0.95),
        "epsilon": 1e-8,
        "weight_decay": args.weight_decay,
        "use_triton": True,
    }
    if dist.get_world_size() == 1:
        adamw_groups = [group for group in groups if group["algorithm"] == "adamw"]
        muon_groups = [group for group in groups if group["algorithm"] == "muon"]
        optimizer = ChainedOptimizer([Muon(adamw_groups, **optimizer_kwargs), OffloadedMuon(muon_groups)])
        if dist.get_rank() == 0:
            momentum_gib = (
                sum(p.numel() * p.element_size() for group in muon_groups for p in group["params"]) / 2**30
            )
            print(f"single-GPU Muon keeps {momentum_gib:.1f} GiB of momentum on CPU", flush=True)
    else:
        optimizer = Muon(groups, **optimizer_kwargs)
    if dist.get_rank() == 0:
        summaries: dict[tuple[str, float, float, bool], tuple[int, int]] = {}
        for group in optimizer.param_groups:
            key = (group["algorithm"], group["base_lr"], group["weight_decay"], group["scheduled"])
            group_count, tensor_count = summaries.get(key, (0, 0))
            summaries[key] = (group_count + 1, tensor_count + len(group["params"]))
        for (algorithm, lr, weight_decay, scheduled), (group_count, tensor_count) in summaries.items():
            print(
                f"optimizer group algorithm={algorithm} groups={group_count} tensors={tensor_count} "
                f"lr={lr:.6g} wd={weight_decay} scheduled={scheduled}",
                flush=True,
            )
    return optimizer


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


def copy_checkpoint_sidecars(base_model: Path, output: Path, params: dict) -> None:
    """params.json and the tokenizer file(s) next to model.pth, so the directory serves as a checkpoint."""
    (output / "params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    spec = params.get("model", params).get("text_tokenizer_spec") or {}
    tokenizer_path = spec.get("path")
    if tokenizer_path and not str(tokenizer_path).startswith("hf://"):
        source = base_model / tokenizer_path
        if not source.is_absolute() and source.is_file():
            destination = output / tokenizer_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def save_for_public_inference(
    model: Zonos3LM,
    params: dict,
    output: Path,
    args: argparse.Namespace,
    rank: int,
    lora_config: LoraConfig | None = None,
    targeted_modules: list[str] | None = None,
) -> None:
    if lora_config is None and not args.export_merged:
        if rank == 0:
            print("checkpoint export disabled", flush=True)
        dist.barrier()
        return
    adapter_only = lora_config is not None and not args.export_merged
    state = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=adapter_only),
    )
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if lora_config is not None:
            adapter = adapter_state_dict(state)
            torch.save(adapter, output / "adapter_model.pt")
            adapter_config = lora_config.to_dict() | {
                "base_model": BASE_MODEL_REPO,
                "targeted_modules": targeted_modules or [],
                "adapter_tensors": len(adapter),
                "adapter_parameters": sum(value.numel() for value in adapter.values()),
            }
            (output / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")
            if args.export_merged:
                state = merge_lora_state_dict(state, lora_config)
        if args.export_merged:
            torch.save(training_to_release_state(state), output / "model.pth")
            copy_checkpoint_sidecars(args.base_model, output, params)
        else:
            (output / "params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
        (output / "training_args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")
        export_kind = "adapter-only" if adapter_only else "public-inference"
        print(f"saved {export_kind} checkpoint to {output}", flush=True)
    dist.barrier()


def save_adapter_snapshot(model: Zonos3LM, output: Path, step: int, rank: int) -> None:
    state = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True)
    )
    if rank == 0:
        snapshot_dir = output / "adapter_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        torch.save(adapter_state_dict(state), snapshot_dir / f"step-{step:06d}.pt")
        print(f"saved adapter snapshot at step {step}", flush=True)
    dist.barrier()


def _torch_optimizers(optimizer) -> list[torch.optim.Optimizer]:
    if isinstance(optimizer, ChainedOptimizer):
        return list(optimizer.optimizers)
    if isinstance(optimizer, torch.optim.Optimizer):
        return [optimizer]
    return []  # LoraMuon keeps its own momentum; its runs are not resumable


def latest_train_state(run_output: Path) -> Path | None:
    root = run_output / "train_state"
    if not root.is_dir():
        return None
    states = sorted(p for p in root.iterdir() if p.is_dir() and (p / "meta.json").is_file())
    return states[-1] if states else None


def save_train_state(model, optimizer, run_output: Path, step: int, epoch: int, consumed: int, args, rank: int) -> None:
    """Model + optimizer state (torch.distributed.checkpoint) and the loader position, for --resume."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    optimizers = _torch_optimizers(optimizer)
    path = run_output / "train_state" / f"step-{step:07d}"
    model_state, optim_state = get_state_dict(model, optimizers)
    dcp.save({"model": model_state, "optim": optim_state}, checkpoint_id=str(path))
    if rank == 0:
        meta = {"step": step, "epoch": epoch, "consumed": consumed, "args": vars(args)}
        (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
        states = sorted(p for p in path.parent.iterdir() if p.is_dir() and (p / "meta.json").is_file())
        for old in states[: -max(1, args.keep_checkpoints)]:
            shutil.rmtree(old, ignore_errors=True)
        print(f"saved training state at step {step} to {path}", flush=True)
    dist.barrier()


def load_train_state(model, optimizer, path: Path) -> dict:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    optimizers = _torch_optimizers(optimizer)
    model_state, optim_state = get_state_dict(model, optimizers)
    state = {"model": model_state, "optim": optim_state}
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(model, optimizers, model_state_dict=state["model"], optim_state_dict=state["optim"])
    return json.loads((path / "meta.json").read_text(encoding="utf-8"))


def split_dataset(dataset: EncodedDataset, val_fraction: float, seed: int) -> tuple[Subset, Subset | None]:
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    val_count = int(round(len(indices) * val_fraction))
    if val_fraction > 0.0:
        val_count = max(1, val_count)
    val_indices = sorted(indices[:val_count])
    train_indices = sorted(indices[val_count:])
    if not train_indices:
        raise ValueError("Validation split leaves no training examples")
    return Subset(dataset, train_indices), (Subset(dataset, val_indices) if val_indices else None)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(model: Zonos3LM, loader: DataLoader, max_batches: int, device: torch.device) -> float:
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        _, loss, _ = model(**batch)
        valid_tokens = (batch["target"] != -100).sum().to(torch.float64)
        loss_sum += loss.detach().to(torch.float64) * valid_tokens
        token_count += valid_tokens
    totals = torch.stack((loss_sum, token_count))
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    model.train()
    if totals[1].item() == 0:
        raise RuntimeError("Validation produced no target tokens")
    return (totals[0] / totals[1]).item()


def write_metric(output: Path, metric: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metric, sort_keys=True) + "\n")


def load_tokenizer(base_model: Path, params: dict):
    spec = params.get("model", params).get("text_tokenizer_spec")
    if not spec:
        return None
    path = Path(spec["path"])
    if not path.is_absolute():
        path = base_model / path
    if not path.is_file():
        return None
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA or ROCm GPU is required. Install the matching environment with "
            "`bash setup.sh nvidia` or `bash setup.sh amd`."
        )
    from datetime import timedelta

    dist.init_process_group("nccl", timeout=timedelta(minutes=float(os.environ.get("ZONOS3_DIST_TIMEOUT_MIN", "30"))))
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    if rank == 0:
        accelerator = f"AMD ROCm {torch.version.hip}" if torch.version.hip else f"NVIDIA CUDA {torch.version.cuda}"
        print(f"accelerator={accelerator} torch={torch.__version__} device={torch.cuda.get_device_name(device)}", flush=True)

    params = load_params(args.base_model)
    model_params = params.get("model", params)
    if args.max_seq_len > int(model_params["max_seqlen"]):
        raise ValueError(f"--max-seq-len {args.max_seq_len} exceeds checkpoint limit {model_params['max_seqlen']}")
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))

    if rank == 0:
        print("building the release-compatible ZONOS3-Mini model", flush=True)
    model_args = release_model_args(params, args)
    with torch.device(device):
        model = Zonos3LM(model_args)
    lora_config = None
    targeted_modules = None
    if args.finetune_method == "lora":
        lora_config = LoraConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            targets=args.lora_targets,
            layer_start=args.lora_layer_start,
            layer_end=args.lora_layer_end,
        )
        targeted_modules = apply_lora(model, lora_config)
        if rank == 0:
            adapter_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"LoRA targets={args.lora_targets} modules={len(targeted_modules)} rank={args.lora_rank} "
                f"alpha={args.lora_alpha:g} dropout={args.lora_dropout:g} "
                f"layers={args.lora_layer_start}:{args.lora_layer_end} adapters={adapter_parameters:,}",
                flush=True,
            )
    model = apply_fsdp(model, mesh)
    load_release_weights(model, args.base_model / "model.pth", rank, allow_missing_lora=lora_config is not None)
    model.set_quantile_balancing(bool(args.qb_update))
    # Dion's Muon compiles its update per parameter shape (fullgraph): the default recompile limit of 8 is too small
    # for the 77 Muon matrices even without --compile.
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.capture_scalar_outputs = True
    if args.compile:
        model.forward = torch.compile(model.forward, backend="inductor", dynamic=True)
        if rank == 0:
            print("torch.compile enabled; GDN=FLA kernels; attention=PyTorch SDPA; MoE=PyTorch expert loop", flush=True)

    token_config = TokenConfig.from_params(params)
    dataset = EncodedDataset(
        args.data, config=token_config, max_seq_len=args.max_seq_len, tokenizer=load_tokenizer(args.base_model, params)
    )
    train_dataset, val_dataset = split_dataset(dataset, args.val_fraction, args.split_seed)
    policy = DrawPolicy(
        p_embedding=args.p_embedding,
        p_reference=args.p_reference,
        embedding_other_clip=args.embedding_other_clip,
        condition_dropout=args.condition_dropout,
        reference_min_seconds=args.reference_min_seconds,
        reference_max_seconds=args.reference_max_seconds,
    )
    if rank == 0:
        speakers = len(dataset.by_speaker)
        print(
            f"dataset train_examples={len(train_dataset)} validation_examples={0 if val_dataset is None else len(val_dataset)} "
            f"speakers={speakers} context_dropped={dataset.context_dropped_count} "
            f"split_seed={args.split_seed} training_seed={args.seed}",
            flush=True,
        )
    sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=False
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=PackingCollator(dataset, policy, args.max_seq_len, seed=args.seed),
    )
    val_loader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=min(args.num_workers, 2),
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=PackingCollator(dataset, policy, args.max_seq_len, deterministic=True, seed=args.split_seed),
        )
    optimizer = build_optimizer(model, args, mesh)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    run_output = args.output.parent
    resumable = args.finetune_method == "full" or isinstance(optimizer, torch.optim.Optimizer)
    start_step, epoch, consumed = 0, 0, 0
    state_path = latest_train_state(run_output) if args.resume else None
    if args.resume and state_path is None and rank == 0:
        print("no training state to resume from; starting from the release weights", flush=True)
    if state_path is not None:
        if not resumable:
            raise RuntimeError("--resume needs a torch optimizer (full fine-tunes, LoRA with AdamW)")
        meta = load_train_state(model, optimizer, state_path)
        start_step, epoch, consumed = int(meta["step"]), int(meta["epoch"]), int(meta["consumed"])
        if rank == 0:
            print(f"resumed from {state_path}: step {start_step}, epoch {epoch}, {consumed} batches into it", flush=True)
        if start_step >= args.steps:
            if rank == 0:
                print(f"run already complete at step {start_step} of {args.steps}; the checkpoint in {args.output} is final", flush=True)
            dist.destroy_process_group()
            return
    if rank == 0:
        run_output.mkdir(parents=True, exist_ok=True)
        if state_path is None:
            (run_output / "metrics.jsonl").write_text("", encoding="utf-8")

    last_val_loss = None
    if val_loader is not None and state_path is None:
        last_val_loss = evaluate(model, val_loader, args.eval_batches, device)
        if rank == 0:
            print(f"validation step=0 loss={last_val_loss:.4f}", flush=True)
            write_metric(run_output, {"step": 0, "validation_loss": last_val_loss})

    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(consumed):  # replay the loader position of a resumed epoch (collation only)
        try:
            next(iterator)
        except StopIteration:
            break
    for step in range(start_step, args.steps):
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        total_tokens = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
                consumed += 1
            except StopIteration:
                epoch += 1
                consumed = 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch_to_device(batch, device)
            _, lm_loss, auxiliary_loss = model(**batch)
            ((lm_loss + auxiliary_loss) / args.grad_accum).backward()
            total_loss += lm_loss.detach().float() / args.grad_accum
            total_tokens += float((batch["target"] != -100).any(dim=-1).sum())

        # A parameter a micro-batch never touched (the speaker projection when no draw had an embedding row) has no
        # gradient on that rank; the distributed optimizer needs the same parameter set on every rank, so give it zeros.
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        if rank == 0 and os.environ.get("ZONOS3_GRAD_DEBUG"):
            norms = []
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    grad = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
                    norms.append((float(grad.float().norm()), name))
            norms.sort(reverse=True)
            print("grad norms: " + "  ".join(f"{name}={value:.1f}" for value, name in norms[:8]), flush=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], args.grad_clip, foreach=True
        )
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier if group.get("scheduled", True) else group["base_lr"]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if args.qb_update:
            model.quantile_balancing_step(ema=args.qb_ema)

        dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
        step_number = step + 1
        should_evaluate = (
            val_loader is not None
            and args.eval_every > 0
            and (step_number % args.eval_every == 0 or step_number == args.steps)
        )
        if should_evaluate:
            last_val_loss = evaluate(model, val_loader, args.eval_batches, device)
        if rank == 0:
            memory = torch.cuda.max_memory_allocated(device) / 2**30
            print(
                f"step={step_number}/{args.steps} loss={total_loss.item():.4f} frames={int(total_tokens.item())} "
                f"grad_norm={grad_norm.item():.3f} lr_mult={multiplier:.6f} memory={memory:.1f}GiB"
                + (f" val={last_val_loss:.4f}" if should_evaluate else ""),
                flush=True,
            )
            metric = {
                "step": step_number,
                "training_loss": total_loss.item(),
                "supervised_frames": int(total_tokens.item()),
                "gradient_norm": grad_norm.item(),
                "lr_multiplier": multiplier,
            }
            if should_evaluate:
                metric["validation_loss"] = last_val_loss
            write_metric(run_output, metric)
        final = step_number == args.steps
        if args.save_every > 0 and (step_number % args.save_every == 0 or final):
            if lora_config is not None and not final:  # the final adapter is exported below
                save_adapter_snapshot(model, args.output, step_number, rank)
            if resumable and lora_config is None:  # the final state makes a chained --resume window a no-op
                save_train_state(model, optimizer, run_output, step_number, epoch, consumed, args, rank)

    save_for_public_inference(
        model, params, args.output, args, rank, lora_config=lora_config, targeted_modules=targeted_modules
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
