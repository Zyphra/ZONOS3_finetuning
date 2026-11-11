"""The real release checkpoint: strict load into the trainer's model, a CPU forward, and (when the ``zonos3``
package is importable) logits parity with the plain-PyTorch reference ``zonos3.reference.lm.Zonos3LM``.

Set ZONOS3_BASE_MODEL to the checkpoint directory (model.pth, params.json); the tests skip without it.
"""

import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import TokenConfig, encode_example  # noqa: E402
from model import Zonos3Args, Zonos3LM  # noqa: E402

BASE = os.environ.get("ZONOS3_BASE_MODEL", "artifacts/base_model")
pytestmark = pytest.mark.skipif(not Path(BASE, "model.pth").is_file(), reason="no release checkpoint at ZONOS3_BASE_MODEL")


def load_trainer_model(dtype=torch.bfloat16) -> Zonos3LM:
    params = json.loads(Path(BASE, "params.json").read_text())
    args = Zonos3Args.from_params(params, gdn_backend="torch")
    with torch.device("meta"):
        model = Zonos3LM(args)
    model.to_empty(device="cpu")
    state = torch.load(Path(BASE, "model.pth"), map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True, assign=True)
    for name, parameter in model.named_parameters():
        parameter.data = parameter.data.to(dtype)
    return model.eval()


def prompt(config: TokenConfig):
    torch.manual_seed(0)
    codes = torch.randint(0, config.codebook_size, (6, config.n_codebooks))
    example = {"codes": codes, "text_ids": [100, 2000, 300, 4000]}
    tokens, target, speaker_position, condition_positions = encode_example(
        example, config, speaker_row=True, condition_ids=[3, 4, 12]
    )
    return tokens, target, speaker_position, condition_positions


def test_strict_load_and_forward():
    model = load_trainer_model()
    params = json.loads(Path(BASE, "params.json").read_text())
    config = TokenConfig.from_params(params)
    tokens, target, speaker_position, condition_positions = prompt(config)
    torch.manual_seed(1)
    embedding = torch.randn(1, model.args.speaker_embedding_dim)
    with torch.no_grad():
        logits, loss, _ = model(
            tokens,
            target,
            torch.tensor([0, tokens.shape[0]], dtype=torch.int32),
            speaker_emb_values=embedding,
            speaker_token_positions=torch.tensor([speaker_position]),
            condition_values=torch.tensor([[3, 4, 12]]),
            condition_token_positions=torch.tensor([condition_positions]),
        )
    assert logits.shape == (tokens.shape[0], config.n_codebooks, config.codebook_size + 2)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    # a random-code "utterance" is far from the model's distribution: the loss must still be a sane cross entropy
    assert 0.0 < float(loss) < 20.0


def test_parity_with_reference_lm():
    """Set ZONOS3_PARITY_DTYPE=bf16 on a small machine (two copies of the weights in float32 need 7 GB)."""
    zonos3 = pytest.importorskip("zonos3.reference.lm")
    dtype = torch.bfloat16 if os.environ.get("ZONOS3_PARITY_DTYPE", "f32") == "bf16" else torch.float32
    model = load_trainer_model(dtype)
    reference = zonos3.Zonos3LM.load(BASE, device="cpu", dtype=dtype)
    params = json.loads(Path(BASE, "params.json").read_text())
    config = TokenConfig.from_params(params)
    tokens, _, speaker_position, condition_positions = prompt(config)
    torch.manual_seed(1)
    embedding = torch.randn(model.args.speaker_embedding_dim)
    with torch.no_grad():
        ours, _, _ = model(
            tokens,
            None,
            torch.tensor([0, tokens.shape[0]], dtype=torch.int32),
            speaker_emb_values=embedding[None],
            speaker_token_positions=torch.tensor([speaker_position]),
            condition_values=torch.tensor([[3, 4, 12]]),
            condition_token_positions=torch.tensor([condition_positions]),
        )
        theirs = reference(
            tokens,
            zonos3.LMCache(),
            speaker_rows={speaker_position: embedding},
            condition_rows={p: v for p, v in zip(condition_positions, [3, 4, 12])},
        ).logits
    ours, theirs = ours.float(), theirs.float()
    agreement = (ours.argmax(-1) == theirs.argmax(-1)).float().mean().item()
    print(f"max |diff| {float((ours - theirs).abs().max()):.4f}, argmax agreement {agreement:.3f}")
    if dtype == torch.float32:
        torch.testing.assert_close(ours, theirs, atol=2e-3, rtol=2e-3)
    else:  # bf16 accumulation order differs between the packed and the token-by-token paths
        torch.testing.assert_close(ours, theirs, atol=0.35, rtol=0.05)
        assert agreement > 0.95
