"""Training sampler: every batch comes from one source dataset, chosen by its `sample_ratio`.

Items are seeded by their batch index, so a resumed run continues the same data stream.
"""

from __future__ import annotations

import contextlib
import hashlib
import random
from collections.abc import Iterator, Sequence
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# Key under which each loaded sample carries its batch seed to the collate function.
BATCH_SEED_KEY = "_batch_seed"


class SampleRequest(NamedTuple):
    sample_index: int
    item_seed: int
    batch_seed: int


def stable_seed(*parts: object) -> int:
    """Stable 63-bit seed from arbitrary scalar parts."""
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "little") & ((1 << 63) - 1)


@contextlib.contextmanager
def rng_context(seed: int) -> Iterator[None]:
    """Temporarily seed the Python, NumPy, and torch CPU global RNGs."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed & 0xFFFF_FFFF)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


class SeededDataset(Dataset):
    """Loads each `SampleRequest` under its item seed and tags it with its batch seed."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, request: SampleRequest) -> dict:
        with rng_context(request.item_seed):
            sample = self.dataset[request.sample_index]
        sample[BATCH_SEED_KEY] = request.batch_seed
        return sample


class MixedBatchSampler(Sampler[list[SampleRequest]]):
    """Yield single-source batches of `SampleRequest`s from a `ConcatDataset`.

    The source of each batch is drawn with the same RNG on every rank, so a DDP global
    batch comes from one dataset. Each source is a shuffled, rank-partitioned pool of
    indices that is reshuffled when exhausted.
    """

    def __init__(
        self,
        dataset_lengths: Sequence[int],
        sample_ratios: Sequence[float],
        batch_size: int,
        num_batches: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        start_batch: int = 0,
    ) -> None:
        self.dataset_lengths = list(dataset_lengths)
        self.sample_ratios = list(sample_ratios)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.start_batch = start_batch
        self.epoch = 0
        self.offsets = np.cumsum([0] + self.dataset_lengths[:-1]).tolist()

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _new_pool(self, source: int, rng: random.Random) -> list[int]:
        # Datasets with fewer samples than ranks are shared by all ranks.
        length = self.dataset_lengths[source]
        pool = list(range(self.rank, length, self.world_size)) or list(range(length))
        rng.shuffle(pool)
        return pool

    def _requests(self, batch: list[int], batch_idx: int) -> list[SampleRequest]:
        batch_seed = stable_seed(self.seed, "batch", self.epoch, batch_idx)
        return [
            SampleRequest(
                sample_index=index,
                item_seed=stable_seed(
                    self.seed, "item", self.epoch, batch_idx, self.rank, pos, index
                ),
                batch_seed=batch_seed,
            )
            for pos, index in enumerate(batch)
        ]

    def __iter__(self) -> Iterator[list[SampleRequest]]:
        source_rng = random.Random(self.seed + self.epoch * 1_000_003)
        pool_rng = random.Random(self.seed + self.epoch * 1_000_003 + self.rank * 9_176)
        sources = list(range(len(self.dataset_lengths)))
        pools = {i: self._new_pool(i, pool_rng) for i in sources}
        cursors = {i: 0 for i in sources}

        # Batches before `start_batch` are drawn and skipped to advance the RNGs.
        for batch_idx in range(self.start_batch + self.num_batches):
            source = source_rng.choices(sources, weights=self.sample_ratios, k=1)[0]
            batch = []
            while len(batch) < self.batch_size:
                pool, cursor = pools[source], cursors[source]
                take = min(self.batch_size - len(batch), len(pool) - cursor)
                batch.extend(
                    self.offsets[source] + i for i in pool[cursor : cursor + take]
                )
                cursors[source] = cursor + take
                if len(batch) < self.batch_size:
                    pools[source] = self._new_pool(source, pool_rng)
                    cursors[source] = 0

            if batch_idx >= self.start_batch:
                yield self._requests(batch, batch_idx)
