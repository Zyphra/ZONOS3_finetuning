#!/usr/bin/env python3
"""Merge a LoRA adapter into the public ZONOS3-Mini checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from finetune import copy_checkpoint_sidecars, release_to_training_state, training_to_release_state
from lora import LoraConfig, merge_lora_state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-file", type=Path, help="Defaults to ADAPTER/adapter_model.pt; may point to a snapshot.")
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.adapter_scale < 0:
        parser.error("--adapter-scale must be non-negative")

    config_payload = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    config = LoraConfig(
        rank=int(config_payload["rank"]),
        alpha=float(config_payload["alpha"]) * args.adapter_scale,
        dropout=float(config_payload.get("dropout", 0.0)),
        targets=str(config_payload["targets"]),
        layer_start=config_payload.get("layer_start"),
        layer_end=config_payload.get("layer_end"),
    )
    adapter_file = args.adapter_file or args.adapter / "adapter_model.pt"

    base = torch.load(args.base_model / "model.pth", map_location="cpu", mmap=True, weights_only=True)
    state = release_to_training_state(base)
    adapter = torch.load(adapter_file, map_location="cpu", weights_only=True)
    overlap = set(state).intersection(adapter)
    if overlap:
        raise RuntimeError(f"Adapter unexpectedly replaces base tensors: {sorted(overlap)[:5]}")
    state.update(adapter)
    merged = training_to_release_state(merge_lora_state_dict(state, config))

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output / "model.pth")
    params = json.loads((args.base_model / "params.json").read_text(encoding="utf-8"))
    copy_checkpoint_sidecars(args.base_model, args.output, params)
    (args.output / "merge_provenance.json").write_text(
        json.dumps(
            {
                "base_model": str(args.base_model.resolve()),
                "adapter": str(args.adapter.resolve()),
                "adapter_file": str(adapter_file.resolve()),
                "adapter_scale": args.adapter_scale,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"merged {len(adapter)} adapter tensors into {args.output}")


if __name__ == "__main__":
    main()
