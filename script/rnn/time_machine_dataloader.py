from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch.utils.data import DataLoader, Dataset

TIME_MACHINE_URL = "https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"


def read_time_machine(root: str | Path = "data") -> list[str]:
    """Download (once), read and clean the corpus as in D2L §8.2."""
    data_dir = Path(root)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "timemachine.txt"
    if not path.exists():
        urlretrieve(TIME_MACHINE_URL, path)
    with path.open("r", encoding="utf-8") as file:
        return [
            re.sub(r"[^A-Za-z]+", " ", line).strip().lower()
            for line in file
        ]


class CharVocab:
    """Character vocabulary, with <unk> at index 0 (same convention as D2L)."""

    def __init__(self, lines: list[str]):
        counts = Counter(char for line in lines for char in line)
        self.idx_to_token = ["<unk>"] + [
            token for token, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        ]
        self.token_to_idx = {token: i for i, token in enumerate(self.idx_to_token)}

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def __getitem__(self, token: str) -> int:
        return self.token_to_idx.get(token, 0)

    def to_tokens(self, ids: int | list[int] | torch.Tensor):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            return self.idx_to_token[ids]
        return [self.idx_to_token[int(i)] for i in ids]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        return "".join(self.to_tokens(ids))


def load_corpus_time_machine(
        max_tokens: int = 10000,
        root: str | Path = "data",
) -> tuple[torch.Tensor, CharVocab]:
    """Return 1D token IDs and vocabulary; vocab is built from ALL text."""
    lines = read_time_machine(root)
    vocab = CharVocab(lines)
    ids = [vocab[char] for line in lines for char in line]
    if max_tokens > 0:
        ids = ids[:max_tokens]
    return torch.tensor(ids, dtype=torch.long), vocab


class TimeMachineDataset(Dataset):
    """Return one (X, Y) pair of shape [num_steps].

    mode='random': each item is a nonoverlapping window; DataLoader shuffles items.
    mode='sequential': items are indexed in batch-major order so that each
    successive batch continues every row's sequence from the previous batch.
    """

    def __init__(
            self,
            corpus: torch.Tensor | list[int],
            batch_size: int,