"""Safe parallel-map helper (spec section 23).

Parallelism is only ever applied to independent units of work (per-symbol,
per-day precompute, PDF/cache generation, RF dataset chunks, RF training). Any
path that carries trading state across time must stay sequential -- this helper
preserves input order in its output so results are identical to a sequential run.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Iterable, List, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def resolve_workers(max_workers: int) -> int:
    if max_workers and max_workers > 0:
        return int(max_workers)
    return max(1, os.cpu_count() or 1)


def parallel_map(
    func: Callable[[T], R],
    items: Sequence[T],
    max_workers: int = 0,
    use_processes: bool = False,
) -> List[R]:
    """Map ``func`` over ``items`` preserving order.

    Falls back to a sequential map when only one worker is available or a single
    item is supplied, which keeps results deterministic and debuggable.
    """
    items = list(items)
    workers = resolve_workers(max_workers)
    if workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]

    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with executor_cls(max_workers=workers) as executor:
        # executor.map preserves the input ordering of results.
        return list(executor.map(func, items))


def chunked(items: Iterable[T], n_chunks: int) -> List[List[T]]:
    """Split ``items`` into at most ``n_chunks`` contiguous chunks (order kept)."""
    items = list(items)
    if n_chunks <= 1 or len(items) <= 1:
        return [items]
    size = (len(items) + n_chunks - 1) // n_chunks
    return [items[i : i + size] for i in range(0, len(items), size)]
