#!/usr/bin/env python3
"""Write a synthetic ``preprocess.py``-style dataset (random codes, embeddings and text ids) for trainer smoke tests.

    python tests/make_synthetic_dataset.py --base-model artifacts/base_model --output data/processed/synthetic.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=Path("artifacts/base_model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=24)
    parser.add_argument("--speakers", type=int, default=3)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    model = json.loads((args.base_model / "params.json").read_text())["model"]
    torch.manual_seed(args.seed)
    counts = model.get("condition_bucket_counts") or [model["condition_n_buckets"]] * model["condition_count"]
    examples = []
    for index in range(args.examples):
        frames = int(torch.randint(args.min_frames, args.max_frames + 1, ()))
        text_len = max(3, frames // 4)
        examples.append(
            {
                "codes": torch.randint(0, model["codebook_size"], (frames, model["n_codebooks"]), dtype=torch.int16),
                "text": "synthetic",
                "text_ids": torch.randint(4, model["text_vocab"] - 1, (text_len,)).tolist(),
                "speaker": f"speaker{index % args.speakers}",
                "speaker_embedding": torch.nn.functional.normalize(torch.randn(model["speaker_embedding_dim"]), dim=0),
                "conditions": [int(torch.randint(0, c, ())) for c in counts],
                "duration": frames / 15.0,
                "segment_id": index,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"examples": examples, "metadata": {"synthetic": True}}, args.output)
    print(f"wrote {len(examples)} synthetic examples to {args.output}")


if __name__ == "__main__":
    main()
