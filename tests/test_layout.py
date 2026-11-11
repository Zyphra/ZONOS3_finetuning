"""The packed-sequence layout of the training loader."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import DrawPolicy, EncodedDataset, PackingCollator, TokenConfig, encode_example, shear  # noqa: E402

CONFIG = TokenConfig(n_codebooks=3, codebook_size=8, eoa_id=8, audio_pad_id=9, text_vocab=20, condition_bucket_counts=(2, 2, 3))


def example(frames: int, text_ids=(5, 6, 7), speaker="a"):
    return {
        "codes": torch.randint(0, 8, (frames, 3), dtype=torch.int16),
        "text_ids": list(text_ids),
        "speaker": speaker,
        "speaker_embedding": torch.randn(4),
        "conditions": [1, 0, 2],
        "segment_id": frames,
    }


def test_shear_delays_each_codebook_by_its_index():
    codes = torch.arange(18).view(6, 3)  # the caller already appended the n_codebooks - 1 tail rows
    delayed = shear(codes, -1)
    assert delayed.shape == (6, 3)
    for cb in range(3):
        assert delayed[cb:, cb].tolist() == codes[: 6 - cb, cb].tolist()
        assert (delayed[:cb, cb] == -1).all()


def test_embedding_draw_layout():
    ex = example(4)
    tokens, target, speaker_position, condition_positions = encode_example(
        ex, CONFIG, speaker_row=True, condition_ids=[1, 0, 2]
    )
    text_len = len(ex["text_ids"]) + 2
    # speaker row, 3 condition rows, text, audio + EOA + delay tail, minus the shifted last row
    assert tokens.shape == (1 + 3 + text_len + 4 + 1 + 2 - 1, 4)
    assert speaker_position == 0 and condition_positions == [1, 2, 3]
    assert (tokens[:4, :3] == CONFIG.audio_pad_id).all() and (tokens[:4, 3] == CONFIG.text_vocab).all()
    assert (target[:4] == -100).all()
    assert tokens[4, 3] == 2 and tokens[4 + text_len - 1, 3] == 3  # BOS ... EOS
    assert (tokens[4:, 3][text_len:] == CONFIG.text_vocab).all()
    first_audio = 4 + text_len
    assert tokens[first_audio, 0] == ex["codes"][0, 0]  # codebook 0 starts right after the text
    assert tokens[first_audio, 1] == CONFIG.audio_pad_id  # codebook 1 is delayed by one row
    assert target[first_audio - 1, 0] == ex["codes"][0, 0]  # the row before predicts the first frame
    assert (target[first_audio - 1, 1:] == -100).all()
    assert (target == CONFIG.eoa_id).sum() == 3  # one EOA per codebook
    assert (target == CONFIG.audio_pad_id).sum() == 0


def test_reference_draw_layout():
    ex = example(4)
    reference = torch.randint(0, 8, (5, 3), dtype=torch.int16)
    tokens, target, speaker_position, condition_positions = encode_example(
        ex, CONFIG, speaker_row=False, condition_ids=[2, 2, 3], reference_codes=reference
    )
    prefix = 5 + 3 - 1
    assert speaker_position is None and condition_positions == [prefix, prefix + 1, prefix + 2]
    assert (target[:prefix] == -100).all()
    assert tokens[0, 0] == reference[0, 0] and tokens[prefix - 1, 2] == reference[4, 2]
    assert (tokens[:prefix, 3] == CONFIG.text_vocab).all()
    with pytest.raises(ValueError):
        encode_example(ex, CONFIG, speaker_row=True, condition_ids=[0, 0, 0], reference_codes=reference)


def test_collator_packs_and_pads(tmp_path):
    examples = [example(6, speaker="a"), example(7, speaker="a"), example(5, speaker="b")]
    path = tmp_path / "data.pt"
    torch.save({"examples": examples, "metadata": {}}, path)
    dataset = EncodedDataset(path, config=CONFIG, max_seq_len=64)
    assert dataset.siblings(0) == [1] and dataset.siblings(2) == []
    policy = DrawPolicy(p_embedding=0.0, p_reference=1.0, reference_min_seconds=0.1, reference_max_seconds=1.0, frame_rate=15)
    collator = PackingCollator(dataset, policy, max_seq_len=64, deterministic=True, pad_multiple=16)
    batch = collator([0, 1, 2])
    assert batch["token_values"].shape[0] % 16 == 0
    assert batch["cu_seqlens"].dtype == torch.int32 and int(batch["cu_seqlens"][-1]) == batch["token_values"].shape[0]
    assert batch["condition_values"].shape == (3, 3) and batch["condition_token_positions"].shape == (3, 3)
    # speakers a: reference draws (sibling exists); speaker b: falls back to the embedding row
    assert batch["speaker_emb_values"].shape == (1, 4)
    assert int(batch["speaker_token_positions"][0]) == int(batch["cu_seqlens"][2])
    for row in batch["condition_token_positions"]:
        assert (batch["token_values"][row, :3] == CONFIG.audio_pad_id).all()
        assert (batch["target"][row] == -100).all()
    # the padding document carries no targets
    assert (batch["target"][int(batch["cu_seqlens"][-2]) :] == -100).all()
