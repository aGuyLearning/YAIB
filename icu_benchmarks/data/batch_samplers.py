"""Batch samplers for pooled multi-source ICU data (e.g. PooledData stay_id suffixes)."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Iterator, List

from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


def pool_source_id_from_stay_id(stay_id: int) -> int:
    """Map stay_id to a source bucket matching YAIB PooledData suffix scheme.

    PooledData uses ``int(str(stay_id) + repeated_digit)`` with ``repeated_digit = str(int_id) * 4``
    for ``int_id`` in 1..9, i.e. last four decimal digits are identical.

    Returns:
        That digit (1-9) if the pattern matches, else 0 (single-site / unknown).
    """
    s = str(abs(int(stay_id)))
    if len(s) >= 4 and s[-1] == s[-2] == s[-3] == s[-4]:
        return int(s[-1])
    return 0


class HomogeneousSourceBatchSampler(Sampler[List[int]]):
    """Yields batches where every index maps to the same pool source bucket."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        drop_last: bool,
        shuffle: bool,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        group_col = dataset.vars["GROUP"]
        stay_ids = dataset.grouping_df[group_col].unique().to_list()
        self._length = len(stay_ids)
        by_source: dict[int, list[int]] = defaultdict(list)
        for i, sid in enumerate(stay_ids):
            by_source[pool_source_id_from_stay_id(int(sid))].append(i)
        self._by_source = {k: v for k, v in by_source.items() if v}
        self._num_batches = self._count_batches_and_warn()

    def _count_batches_and_warn(self) -> int:
        total = 0
        for src, indices in sorted(self._by_source.items()):
            k = len(indices)
            if self.drop_last:
                if k < self.batch_size:
                    logger.warning(
                        "HomogeneousSourceBatchSampler: source bucket %s has %d stays (< batch_size=%d); "
                        "drop_last=True drops all of them.",
                        src,
                        k,
                        self.batch_size,
                    )
                total += k // self.batch_size
            else:
                total += (k + self.batch_size - 1) // self.batch_size
        return total

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[List[int]]:
        sources = sorted(self._by_source.keys())
        if self.shuffle:
            sources = list(self._by_source.keys())
            random.shuffle(sources)
        batches: list[list[int]] = []
        for src in sources:
            idxs = list(self._by_source[src])
            if self.shuffle:
                random.shuffle(idxs)
            i = 0
            while i < len(idxs):
                chunk = idxs[i : i + self.batch_size]
                i += self.batch_size
                if len(chunk) < self.batch_size and self.drop_last:
                    break
                batches.append(chunk)
        if self.shuffle:
            random.shuffle(batches)
        yield from batches
