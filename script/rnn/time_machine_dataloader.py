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
        num_steps: int,
        mode: str = "sequential",
        offset: int | None = None,
    ):
        if batch_size < 1 or num_steps < 1:
            raise ValueError("batch_size and num_steps must be >= 1")
        if mode not in {"random", "sequential"}:
            raise ValueError("mode must be 'random' or 'sequential'")

        self.corpus = torch.as_tensor(corpus, dtype=torch.long)
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.mode = mode
        if offset is None:
            offset = random.randint(0, num_steps - 1 if mode == "random" else num_steps)
        max_offset = num_steps - 1 if mode == "random" else num_steps
        if not 0 <= offset <= max_offset:
            raise ValueError(f"offset must be in [0, {max_offset}] for {mode} mode")
        self.offset = offset

        if mode == "random":
            # Reserve one token per window for the shifted label Y.
            num_subseqs = max(0, (len(self.corpus) - offset - 1) // num_steps)
            self.starts = [offset + i * num_steps for i in range(num_subseqs)]
            self.n_samples = len(self.starts)
        else:
            # Split the contiguous corpus into batch_size long streams,
            # then read successive chunks from each stream in lockstep.
            usable = max(0, (len(self.corpus) - offset - 1) // batch_size * batch_size)
            self.Xs = self.corpus[offset : offset + usable].reshape(batch_size, -1)
            self.Ys = self.corpus[offset + 1 : offset + 1 + usable].reshape(batch_size, -1)
            self.num_batches = self.Xs.shape[1] // num_steps
            self.n_samples = self.num_batches * batch_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.mode == "random":
            i = self.starts[index]
            X = self.corpus[i : i + self.num_steps]
            Y = self.corpus[i + 1 : i + self.num_steps + 1]
        else:
            row = index % self.batch_size
            batch_index = index // self.batch_size
            start = batch_index * self.num_steps
            stop = start + self.num_steps
            X = self.Xs[row, start:stop]
            Y = self.Ys[row, start:stop]
        return X, Y


def load_time_machine_dataloader(
    batch_size: int = 32,
    num_steps: int = 35,
    use_random_iter: bool = False,
    max_tokens: int = 10000,
    root: str | Path = "data",
    seed: int | None = None,
) -> tuple[DataLoader, CharVocab]:
    """Ready-to-use D2L-style (loader, vocab), relying on torch DataLoader.

    Returns batches X, Y as int64 tensors of shape [batch_size, num_steps].
    'sequential' is suitable for passing RNN hidden state between batches.
    'random' shuffles the order of nonoverlapping windows each epoch.

    Note: this DataLoader chooses the initial offset once per construction;
    recreate the loader to choose a different offset in a later epoch.
    """
    corpus, vocab = load_corpus_time_machine(max_tokens=max_tokens, root=root)
    mode = "random" if use_random_iter else "sequential"
    rng = random.Random(seed)
    offset = rng.randint(0, num_steps - 1 if use_random_iter else num_steps)
    dataset = TimeMachineDataset(
        corpus=corpus,
        batch_size=batch_size,
        num_steps=num_steps,
        mode=mode,
        offset=offset,
    )
    if len(dataset) < batch_size:
        raise ValueError("Corpus is too short for one complete batch; reduce batch_size/num_steps")

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=use_random_iter,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    return loader, vocab


if __name__ == "__main__":
    loader, vocab = load_time_machine_dataloader(
        batch_size=2,
        num_steps=10,
        use_random_iter=False,
        max_tokens=10000,
        seed=42,
    )
    X, Y = next(iter(loader))
    print(f"Vocab size: {len(vocab)}")
    print(f"X.shape={tuple(X.shape)}; Y.shape={tuple(Y.shape)}")
    print("Input: ", repr(vocab.decode(X[0])))
    print("Target:", repr(vocab.decode(Y[0])))
    print("Y is one token to the right of X:",
          bool(torch.equal(X[:, 1:], Y[:, :-1])))
