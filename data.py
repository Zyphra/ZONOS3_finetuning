"""ZONOS3 training examples and packed sequences in the training loader's layout.

One training sequence (rows of ``n_codebooks + 1`` ids, the last column the text id) is, in order:

1. an optional **raw reference prefix**: the codec codes of another clip of the same speaker, each codebook delayed
   by its index and the tail flushed with audio padding, text column = text pad, never supervised;
2. or an optional **speaker row**: one all-pad row whose embedding the model replaces by the projected speaker
   embedding (a draw conditions on a prefix or on an embedding, never both; some draws on neither);
3. the **condition rows**: one all-pad row per condition (loudness, fmax, speaking rate) whose embedding the model
   replaces by the bucket table entry (dropout replaces a bucket by the table's unknown id);
4. the **text**: BOS, the SuperBPE payload ids (+4 offset), EOS, with audio padding in the codebook columns;
5. the **audio**: the codes delayed by their codebook index (the shear), then EOA on every codebook, then the delay
   tail. Targets are the next row's audio columns; padding is never supervised.

``preprocess.py`` writes the examples this module reads: codes ``int16 [frames, 10]``, SuperBPE ``text_ids``, the
``speaker`` id, the 192-d ``speaker_embedding`` and the ``conditions`` bucket ids.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

BOS_ID, EOS_ID = 2, 3
BPE_OFFSET = 4


@dataclass(frozen=True)
class TokenConfig:
    n_codebooks: int = 10
    codebook_size: int = 4096
    eoa_id: int = 4096
    audio_pad_id: int = 4097
    text_vocab: int = 8196
    condition_bucket_counts: tuple[int, ...] = (10, 10, 30)

    @classmethod
    def from_params(cls, params: dict) -> "TokenConfig":
        p = params.get("model", params)
        count = int(p.get("condition_count", 3))
        counts = p.get("condition_bucket_counts") or [int(p.get("condition_n_buckets", 10))] * count
        return cls(
            n_codebooks=int(p["n_codebooks"]),
            codebook_size=int(p["codebook_size"]),
            eoa_id=int(p["eoa_id"]),
            audio_pad_id=int(p["audio_pad_id"]),
            text_vocab=int(p["text_vocab"]),
            condition_bucket_counts=tuple(int(c) for c in counts),
        )

    @property
    def condition_count(self) -> int:
        return len(self.condition_bucket_counts)

    @property
    def unknown_ids(self) -> list[int]:
        return [int(c) for c in self.condition_bucket_counts]


@dataclass(frozen=True)
class DrawPolicy:
    """How a training draw conditions the sequence (the fine-tuning mix of the release recipe)."""

    p_embedding: float = 0.45  # reserved speaker row + MECT embedding
    p_reference: float = 0.45  # raw reference prefix: another clip of the same speaker (the rest: unconditioned)
    embedding_other_clip: float = 0.5  # the embedding of another clip of the same speaker, when one exists
    condition_dropout: float = 0.2  # per condition row -> the unknown bucket
    reference_min_seconds: float = 1.0
    reference_max_seconds: float = 30.0
    frame_rate: float = 15.0

    def __post_init__(self):
        if not 0.0 <= self.p_embedding <= 1.0 or not 0.0 <= self.p_reference <= 1.0:
            raise ValueError("draw probabilities must be in [0, 1]")
        if self.p_embedding + self.p_reference > 1.0 + 1e-9:
            raise ValueError("p_embedding + p_reference must not exceed 1")
        if not 0.0 <= self.condition_dropout <= 1.0:
            raise ValueError("condition_dropout must be in [0, 1]")

    @property
    def reference_min_frames(self) -> int:
        return max(1, int(round(self.reference_min_seconds * self.frame_rate)))

    @property
    def reference_max_frames(self) -> int:
        return max(self.reference_min_frames, int(self.reference_max_seconds * self.frame_rate))


def shear(codes: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Delay codebook ``c`` by ``c`` rows."""
    length, n_codebooks = codes.shape
    padded = codes.new_full((length + n_codebooks - 1, n_codebooks), pad_id)
    padded[n_codebooks - 1 :] = codes
    rows = n_codebooks - 1 + torch.arange(length).unsqueeze(1) - torch.arange(n_codebooks)
    return padded.gather(0, rows)


def text_ids_with_boundaries(payload_ids: list[int]) -> list[int]:
    return [BOS_ID, *payload_ids, EOS_ID]


def payload_ids_from_text(text: str, tokenizer) -> list[int]:
    """SuperBPE payload ids with the training offset (``tokenizers.Tokenizer``)."""
    return [token + BPE_OFFSET for token in tokenizer.encode(text, add_special_tokens=False).ids]


def example_length(example: dict, config: TokenConfig, *, speaker_row: bool, reference_frames: int = 0) -> int:
    """Rows of the packed sequence of ``example`` under the given conditioning."""
    text_rows = len(example["text_ids"]) + 2
    frames = int(example["codes"].shape[0])
    length = text_rows + frames + 1 + config.n_codebooks - 1  # text, audio, EOA, tail; minus the dropped last row
    length += int(speaker_row) + config.condition_count
    if reference_frames:
        length += reference_frames + config.n_codebooks - 1
    return length


def encode_reference(codes: torch.Tensor, config: TokenConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """The raw reference prefix rows and their (unsupervised) targets."""
    audio = codes.long()
    if audio.ndim != 2 or audio.shape[1] != config.n_codebooks or not len(audio):
        raise ValueError("invalid reference codes")
    length = len(audio) + config.n_codebooks - 1
    columns = torch.full((length, config.n_codebooks), config.audio_pad_id, dtype=torch.long)
    columns[: len(audio)] = audio
    tokens = torch.cat((shear(columns, config.audio_pad_id), torch.full((length, 1), config.text_vocab)), dim=-1)
    return tokens, torch.full((length, config.n_codebooks), -100, dtype=torch.long)


def encode_example(
    example: dict,
    config: TokenConfig,
    *,
    speaker_row: bool,
    condition_ids: list[int] | None,
    reference_codes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int | None, list[int]]:
    """One sequence: (token_values [L, C + 1], target [L, C], speaker row position, condition row positions)."""
    audio = example["codes"].long()
    if audio.ndim != 2 or audio.shape[1] != config.n_codebooks:
        raise ValueError(f"expected codes shaped [frames, {config.n_codebooks}], got {tuple(audio.shape)}")
    if audio.numel() and (int(audio.min()) < 0 or int(audio.max()) >= config.codebook_size):
        raise ValueError("codes outside the codebook vocabulary")
    text = torch.tensor(text_ids_with_boundaries(list(example["text_ids"])), dtype=torch.long)
    if int(text.max()) >= config.text_vocab or int(text.min()) < 0:
        raise ValueError("text ids outside the text vocabulary")

    audio_start = text.numel()
    length = audio_start + audio.shape[0] + 1 + config.n_codebooks - 1
    audio_columns = torch.full((length, config.n_codebooks), config.audio_pad_id, dtype=torch.long)
    text_column = torch.full((length,), config.text_vocab, dtype=torch.long)
    text_column[:audio_start] = text
    audio_columns[audio_start : audio_start + audio.shape[0]] = audio
    audio_columns[audio_start + audio.shape[0]] = config.eoa_id
    delayed = shear(audio_columns, config.audio_pad_id)
    token_values = torch.cat((delayed[:-1], text_column[:-1, None]), dim=-1)
    target = delayed[1:].clone()
    target[target == config.audio_pad_id] = -100

    condition_positions: list[int] = []
    if condition_ids is not None:
        count = config.condition_count
        if len(condition_ids) != count:
            raise ValueError(f"expected {count} condition ids, got {len(condition_ids)}")
        slots = torch.full((count, config.n_codebooks + 1), config.audio_pad_id, dtype=torch.long)
        slots[:, -1] = config.text_vocab
        token_values = torch.cat((slots, token_values), dim=0)
        target = torch.cat((torch.full((count, config.n_codebooks), -100, dtype=torch.long), target), dim=0)
        condition_positions = list(range(count))
    speaker_position: int | None = None
    if speaker_row:
        if reference_codes is not None:
            raise ValueError("a draw conditions on a reference prefix or a speaker embedding, never both")
        slot = torch.full((1, config.n_codebooks + 1), config.audio_pad_id, dtype=torch.long)
        slot[:, -1] = config.text_vocab
        token_values = torch.cat((slot, token_values), dim=0)
        target = torch.cat((torch.full((1, config.n_codebooks), -100, dtype=torch.long), target), dim=0)
        speaker_position = 0
        condition_positions = [p + 1 for p in condition_positions]
    if reference_codes is not None:
        prefix_tokens, prefix_target = encode_reference(reference_codes, config)
        token_values = torch.cat((prefix_tokens, token_values), dim=0)
        target = torch.cat((prefix_target, target), dim=0)
        condition_positions = [p + len(prefix_tokens) for p in condition_positions]
    return token_values, target, speaker_position, condition_positions


class EncodedDataset(Dataset):
    """Pre-encoded examples (``preprocess.py``), grouped by speaker for the other-clip draws."""

    def __init__(self, path: str | Path, *, config: TokenConfig, max_seq_len: int | None = None, tokenizer=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.metadata = payload.get("metadata", {})
        self.config = config
        examples = []
        self.context_dropped_count = 0
        for index, example in enumerate(payload["examples"]):
            example = dict(example)
            example.setdefault("segment_id", index)
            if "text_ids" not in example:
                if tokenizer is None:
                    raise ValueError("examples without text_ids need a SuperBPE tokenizer")
                example["text_ids"] = payload_ids_from_text(example["text"], tokenizer)
            if example.get("speaker") is None:
                example["speaker"] = "speaker"
            if "speaker_embedding" in example and example["speaker_embedding"] is not None:
                example["speaker_embedding"] = torch.as_tensor(example["speaker_embedding"], dtype=torch.float32)
            # The sequence must fit with a speaker row; the collator trims any reference prefix to the context.
            if max_seq_len is not None and example_length(example, config, speaker_row=True) > max_seq_len:
                self.context_dropped_count += 1
                continue
            examples.append(example)
        if not examples:
            raise ValueError("no example fits the context limit")
        self.examples = examples
        self.by_speaker: dict[str, list[int]] = {}
        for index, example in enumerate(self.examples):
            self.by_speaker.setdefault(str(example["speaker"]), []).append(index)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> int:
        return index  # the collator reads the examples (and their siblings) from the dataset

    def siblings(self, index: int) -> list[int]:
        speaker = str(self.examples[index]["speaker"])
        return [i for i in self.by_speaker.get(speaker, []) if i != index]


class PackingCollator:
    """Draw the conditioning of each example and pack the batch into one packed sequence set."""

    def __init__(
        self,
        dataset: EncodedDataset,
        policy: DrawPolicy,
        max_seq_len: int,
        *,
        deterministic: bool = False,
        pad_multiple: int = 64,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.config = dataset.config
        self.policy = policy
        self.max_seq_len = int(max_seq_len)
        self.deterministic = deterministic
        self.pad_multiple = int(pad_multiple)
        self.seed = int(seed)

    def _rng(self, index: int) -> random.Random:
        if self.deterministic:
            return random.Random((self.seed * 1_000_003 + int(self.dataset.examples[index]["segment_id"])) & 0xFFFFFFFF)
        return random  # the DataLoader worker's seeded module RNG

    def _conditions(self, example: dict, rng) -> list[int]:
        unknown = self.config.unknown_ids
        ids = example.get("conditions")
        if ids is None:
            return list(unknown)
        ids = [int(v) for v in ids]
        if len(ids) != self.config.condition_count:
            raise ValueError("wrong number of condition ids")
        return [
            unknown[i] if (not 0 <= ids[i] <= unknown[i] or rng.random() < self.policy.condition_dropout) else ids[i]
            for i in range(len(ids))
        ]

    def _reference(self, index: int, rng) -> torch.Tensor | None:
        """Codes of another clip of the same speaker, cropped to the reference budget and the context."""
        siblings = self.dataset.siblings(index)
        if not siblings:
            return None
        other = self.dataset.examples[rng.choice(siblings)]
        codes = other["codes"]
        base = example_length(self.dataset.examples[index], self.config, speaker_row=False)
        budget = self.max_seq_len - base - (self.config.n_codebooks - 1)
        frames = min(int(codes.shape[0]), self.policy.reference_max_frames, budget)
        if frames < self.policy.reference_min_frames:
            return None
        start = 0 if self.deterministic or frames == codes.shape[0] else rng.randrange(0, codes.shape[0] - frames + 1)
        return codes[start : start + frames]

    def _embedding(self, index: int, rng) -> torch.Tensor | None:
        example = self.dataset.examples[index]
        own = example.get("speaker_embedding")
        siblings = self.dataset.siblings(index)
        if siblings and rng.random() < self.policy.embedding_other_clip:
            other = self.dataset.examples[rng.choice(siblings)].get("speaker_embedding")
            if other is not None:
                return other
        return own

    def __call__(self, indices: list[int]) -> dict:
        encoded, lengths = [], []
        speaker_embeddings, speaker_positions = [], []
        condition_values, condition_positions = [], []
        total = 0
        for index in indices:
            rng = self._rng(index)
            example = self.dataset.examples[index]
            draw = rng.random()
            reference = embedding = None
            if draw < self.policy.p_reference:
                reference = self._reference(index, rng)
                if reference is None:  # no usable sibling: fall back to the embedding row
                    embedding = self._embedding(index, rng)
            elif draw < self.policy.p_reference + self.policy.p_embedding:
                embedding = self._embedding(index, rng)
            conditions = self._conditions(example, rng)
            tokens, target, speaker_position, positions = encode_example(
                example,
                self.config,
                speaker_row=embedding is not None,
                condition_ids=conditions,
                reference_codes=reference,
            )
            if tokens.shape[0] > self.max_seq_len:
                raise ValueError(f"packed example length {tokens.shape[0]} exceeds max_seq_len {self.max_seq_len}")
            if embedding is not None:
                speaker_embeddings.append(embedding.to(torch.float32))
                speaker_positions.append(total + speaker_position)
            condition_values.append(conditions)
            condition_positions.append([total + p for p in positions])
            encoded.append((tokens, target))
            lengths.append(tokens.shape[0])
            total += tokens.shape[0]
        if self.pad_multiple > 1 and total % self.pad_multiple:
            n = self.pad_multiple - total % self.pad_multiple
            tokens = torch.full((n, self.config.n_codebooks + 1), self.config.audio_pad_id, dtype=torch.long)
            tokens[:, -1] = self.config.text_vocab
            encoded.append((tokens, torch.full((n, self.config.n_codebooks), -100, dtype=torch.long)))
            lengths.append(n)
            total += n
        batch = {
            "token_values": torch.cat([item[0] for item in encoded]),
            "target": torch.cat([item[1] for item in encoded]),
            "cu_seqlens": torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32),
            "maxlen": max(lengths),
            "condition_values": torch.tensor(condition_values, dtype=torch.long),
            "condition_token_positions": torch.tensor(condition_positions, dtype=torch.long),
        }
        if speaker_embeddings:
            batch["speaker_emb_values"] = torch.stack(speaker_embeddings, dim=0)
            batch["speaker_token_positions"] = torch.tensor(speaker_positions, dtype=torch.long)
        return batch
