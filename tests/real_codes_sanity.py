#!/usr/bin/env python3
"""Teacher-forced loss of the release weights on real codec codes versus random codes (CPU, no codec needed).

    python tests/real_codes_sanity.py --base-model artifacts/base_model \
        --codes path/to/Voice.codes.npy --embedding path/to/Voice.embedding.npy

Real speech codes must score far below uniform random codes (log 4096 = 8.3 nats) even with a placeholder text:
a check of the token layout, the delay pattern and the conditioning rows against the trained weights.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import TokenConfig, encode_example  # noqa: E402
from model import Zonos3Args, Zonos3LM  # noqa: E402


def load_model(base: Path) -> Zonos3LM:
    params = json.loads((base / "params.json").read_text())
    args = Zonos3Args.from_params(params, gdn_backend="torch")
    with torch.device("meta"):
        model = Zonos3LM(args)
    model.to_empty(device="cpu")
    state = torch.load(base / "model.pth", map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def loss_of(model, config, codes, text_ids, embedding, conditions):
    example = {"codes": torch.as_tensor(codes), "text_ids": text_ids}
    tokens, target, speaker_position, condition_positions = encode_example(
        example, config, speaker_row=embedding is not None, condition_ids=conditions
    )
    kwargs = {}
    if embedding is not None:
        kwargs = {"speaker_emb_values": embedding[None], "speaker_token_positions": torch.tensor([speaker_position])}
    with torch.no_grad():
        _, loss, _ = model(
            tokens,
            target,
            torch.tensor([0, tokens.shape[0]], dtype=torch.int32),
            condition_values=torch.tensor([conditions]),
            condition_token_positions=torch.tensor([condition_positions]),
            **kwargs,
        )
    return float(loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=Path("artifacts/base_model"))
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--embedding", type=Path)
    parser.add_argument("--frames", type=int, default=150)
    args = parser.parse_args()
    model = load_model(args.base_model)
    config = TokenConfig.from_params(json.loads((args.base_model / "params.json").read_text()))
    codes = np.load(args.codes).astype(np.int64)[: args.frames]
    embedding = torch.from_numpy(np.load(args.embedding).astype(np.float32)) if args.embedding else None
    unknown = config.unknown_ids
    text = [100, 2000, 300, 4000, 500]  # a placeholder transcript: the audio prior still dominates
    torch.manual_seed(0)
    random_codes = torch.randint(0, config.codebook_size, codes.shape)
    results = {
        "real codes, speaker row, unknown conditions": loss_of(model, config, codes, text, embedding, unknown),
        "real codes, no speaker row, unknown conditions": loss_of(model, config, codes, text, None, unknown),
        "random codes, speaker row, unknown conditions": loss_of(model, config, random_codes, text, embedding, unknown),
    }
    for name, value in results.items():
        print(f"{value:7.3f}  {name}")
    print(f"{math.log(config.codebook_size):7.3f}  uniform over the codebook")
    real, random_loss = results["real codes, speaker row, unknown conditions"], results["random codes, speaker row, unknown conditions"]
    if real >= random_loss - 1.0:
        raise SystemExit("real codes are not scored better than random codes: the layout or the weights are wrong")
    print("ok: real codes score far below random codes")


if __name__ == "__main__":
    main()
