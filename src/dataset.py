"""PyTorch Dataset and DataLoader with sliding-window chunking for Funnel-GPT."""

from __future__ import annotations

from functools import partial
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset


class FunnelDataset(Dataset[tuple[list[int], list[int]]]):
    """Sliding-window next-token prediction dataset for tokenized user sessions."""

    def __init__(
        self,
        encoded_sessions: Sequence[Sequence[int]],
        context_length: int,
        stride: int | None = None,
    ) -> None:
        self.context_length = context_length
        self.stride = stride if stride is not None else max(1, context_length // 2)
        self.samples: list[tuple[list[int], list[int]]] = []

        for seq in encoded_sessions:
            if len(seq) < 2:
                continue
            if len(seq) - 1 <= context_length:
                self.samples.append((list(seq[:-1]), list(seq[1:])))
            else:
                last_start = len(seq) - 1 - context_length
                starts = list(range(0, last_start + 1, self.stride))
                if starts[-1] != last_start:
                    starts.append(last_start)
                for start in starts:
                    inp = list(seq[start : start + context_length])
                    tgt = list(seq[start + 1 : start + context_length + 1])
                    self.samples.append((inp, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.samples[idx]


def _collate_padded(
    batch: list[tuple[list[int], list[int]]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(inp) for inp, _ in batch)
    batch_size = len(batch)
    inputs = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    targets = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for row_idx, (inp, tgt) in enumerate(batch):
        length = len(inp)
        inputs[row_idx, :length] = torch.tensor(inp, dtype=torch.long)
        targets[row_idx, :length] = torch.tensor(tgt, dtype=torch.long)
    return inputs, targets


def create_dataloader(
    encoded_sessions: Sequence[Sequence[int]],
    context_length: int,
    pad_id: int,
    batch_size: int = 64,
    shuffle: bool = True,
    stride: int | None = None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a DataLoader with dynamic batch right-padding."""
    dataset = FunnelDataset(
        encoded_sessions=encoded_sessions,
        context_length=context_length,
        stride=stride,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(_collate_padded, pad_id=pad_id),
    )
