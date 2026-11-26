"""A tiny random ZONOS3 configuration on the CPU: forward/backward, LoRA injection and merge."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import TokenConfig, encode_example  # noqa: E402
from lora import LoraConfig, adapter_state_dict, apply_lora, merge_lora_state_dict  # noqa: E402
from model import Zonos3Args, Zonos3LM  # noqa: E402

ARGS = Zonos3Args(
    n_layers=4,
    dim=64,
    head_dim=16,
    n_heads=4,
    n_kv_heads=2,
    max_seqlen=256,
    ffn_hidden=96,
    moe_hidden=48,
    moe_routed_dim=16,
    n_experts=4,
    moe_topk=2,
    n_codebooks=3,
    codebook_size=8,
    eoa_id=8,
    audio_pad_id=9,
    text_vocab=20,
    speaker_embedding_dim=6,
    condition_bucket_counts=(2, 2, 3),
    gdn_backend="torch",
)
CONFIG = TokenConfig(n_codebooks=3, codebook_size=8, eoa_id=8, audio_pad_id=9, text_vocab=20, condition_bucket_counts=(2, 2, 3))


def batch(seed=0):
    torch.manual_seed(seed)
    items = []
    for frames in (5, 7):
        ex = {"codes": torch.randint(0, 8, (frames, 3)), "text_ids": [4, 5, 6]}
        items.append(encode_example(ex, CONFIG, speaker_row=True, condition_ids=[1, 0, 2]))
    tokens = torch.cat([t for t, _, _, _ in items])
    target = torch.cat([y for _, y, _, _ in items])
    lengths = [t.shape[0] for t, _, _, _ in items]
    cu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    starts = [0, lengths[0]]
    return {
        "token_values": tokens,
        "target": target,
        "cu_seqlens": cu,
        "maxlen": max(lengths),
        "speaker_emb_values": torch.randn(2, 6),
        "speaker_token_positions": torch.tensor(starts),
        "condition_values": torch.tensor([[1, 0, 2], [1, 0, 2]]),
        "condition_token_positions": torch.tensor([[s + 1, s + 2, s + 3] for s in starts]),
    }


def test_state_dict_names_follow_the_release_layout():
    model = Zonos3LM(ARGS)
    keys = set(model.state_dict())
    assert "layers.0.attention.in_proj.weight" in keys and "layers.0.attention.conv.weight" in keys
    assert "layers.3.attention.wkv.weight" in keys and "layers.3.attention.temp" in keys
    assert "layers.0.feed_forward.w_in.weight" in keys
    assert "layers.1.feed_forward.experts.gate_up_proj" in keys and "layers.1.feed_forward.router.balancing_biases" in keys
    assert "block_attnres_queries" in keys and "speaker_projection.bias" in keys and "condition_embeddings.2.weight" in keys
    assert model.state_dict()["layers.3.attention.wkv.weight"].shape == (2, 32, 64)
    assert model.state_dict()["multi_embedder.embedders.3.weight"].shape == (21, 64)


def test_forward_backward_and_loss():
    torch.manual_seed(1)
    model = Zonos3LM(ARGS)
    logits, loss, aux = model(**batch())
    assert logits.shape == (batch()["token_values"].shape[0], 3, 10)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert model.layers[1].feed_forward.router.balancing_biases.requires_grad is False


def test_packing_matches_separate_sequences():
    """Document boundaries isolate the sequences: packed logits equal the per-sequence logits."""
    torch.manual_seed(2)
    model = Zonos3LM(ARGS).eval()
    b = batch()
    with torch.no_grad():
        packed, _, _ = model(**b)
        cu = b["cu_seqlens"].tolist()
        for i, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
            single, _, _ = model(
                b["token_values"][start:end],
                cu_seqlens=torch.tensor([0, end - start], dtype=torch.int32),
                speaker_emb_values=b["speaker_emb_values"][i : i + 1],
                speaker_token_positions=torch.tensor([0]),
                condition_values=b["condition_values"][i : i + 1],
                condition_token_positions=b["condition_token_positions"][i : i + 1] - start,
            )
            torch.testing.assert_close(packed[start:end], single, atol=1e-4, rtol=1e-4)


def test_quantile_balancing_updates_biases():
    torch.manual_seed(3)
    model = Zonos3LM(ARGS)
    model.set_quantile_balancing(True)
    b = batch()
    # tokens * top_k must be divisible by the expert count: pad the packed batch like the collator does
    n = b["token_values"].shape[0]
    pad = (-n) % 4
    tokens = torch.cat([b["token_values"], torch.full((pad, 4), 9)])
    tokens[n:, 3] = 20
    target = torch.cat([b["target"], torch.full((pad, 3), -100)])
    cu = torch.cat([b["cu_seqlens"], torch.tensor([n + pad], dtype=torch.int32)])
    _, loss, _ = model(tokens, target, cu, speaker_emb_values=b["speaker_emb_values"], speaker_token_positions=b["speaker_token_positions"], condition_values=b["condition_values"], condition_token_positions=b["condition_token_positions"])
    loss.backward()
    before = model.layers[1].feed_forward.router.balancing_biases.clone()
    model.quantile_balancing_step(ema=0.0)
    after = model.layers[1].feed_forward.router.balancing_biases
    assert not torch.equal(before, after)
    assert abs(float(after.mean())) < 1e-5


def test_lora_apply_and_merge_round_trip():
    torch.manual_seed(4)
    model = Zonos3LM(ARGS).eval()
    b = batch()
    with torch.no_grad():
        base_logits, _, _ = model(**b)
    config = LoraConfig(rank=4, alpha=8.0, targets="all")
    targeted = apply_lora(model, config)
    assert any(name.endswith(".attention.wkv") for name in targeted)
    assert any(name.endswith(".feed_forward.experts") for name in targeted)
    assert "multi_output" in targeted
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith(("lora_a", "lora_b")) for n in trainable)
    with torch.no_grad():
        zero_logits, _, _ = model(**b)
    torch.testing.assert_close(zero_logits, base_logits)  # B = 0: the adapters start as the identity
    # perturb the adapters, then merge and compare with a fresh model loaded from the merged state dict
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora_b"):
                parameter.normal_(std=0.05)
        adapted_logits, _, _ = model(**b)
    state = model.state_dict()
    adapter = adapter_state_dict(state)
    assert adapter and all(k.endswith(("lora_a", "lora_b")) for k in adapter)
    merged = merge_lora_state_dict(state, config)
    fresh = Zonos3LM(ARGS).eval()
    fresh.load_state_dict(merged, strict=True)
    with torch.no_grad():
        merged_logits, _, _ = fresh(**b)
    torch.testing.assert_close(merged_logits, adapted_logits, atol=2e-3, rtol=2e-3)
    assert not torch.allclose(merged_logits, base_logits)


def test_training_to_release_state_casts_fp32_masters_to_release_dtypes():
    from finetune import training_to_release_state

    model = Zonos3LM(ARGS).float()
    exported = training_to_release_state(model.state_dict())
    assert exported
    for key, value in exported.items():
        if key.endswith("balancing_biases"):
            assert value.dtype == torch.float32, f"{key} should stay float32"
        elif value.is_floating_point():
            assert value.dtype == torch.bfloat16, f"{key} should export as bfloat16"


def test_lr_multiplier_warmup_and_cosine_decay():
    from finetune import lr_multiplier

    assert lr_multiplier(0, total_steps=100, warmup_steps=10, min_ratio=1e-6) == 0.1
    assert lr_multiplier(9, total_steps=100, warmup_steps=10, min_ratio=1e-6) == 1.0
    assert lr_multiplier(10, total_steps=100, warmup_steps=10, min_ratio=1e-6) == 1.0
    assert abs(lr_multiplier(100, total_steps=100, warmup_steps=10, min_ratio=1e-6) - 1e-6) < 1e-12
