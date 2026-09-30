"""Plan fixed-size batches to avoid dynamic-shape GPU recompilation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BatchPlan:
    start: int
    stop: int
    compiled_batch_size: int

    @property
    def item_count(self) -> int:
        return self.stop - self.start

    @property
    def padding_count(self) -> int:
        return self.compiled_batch_size - self.item_count


def plan_static_batches(item_count: int, buckets: tuple[int, ...]) -> tuple[BatchPlan, ...]:
    if item_count < 0:
        raise ValueError("item_count must be non-negative")
    if not buckets or any(bucket <= 0 for bucket in buckets):
        raise ValueError("buckets must contain positive values")
    normalized = tuple(sorted(set(buckets)))
    if item_count == 0:
        return ()

    plans: list[BatchPlan] = []
    start = 0
    largest = normalized[-1]
    while item_count - start > largest:
        plans.append(BatchPlan(start=start, stop=start + largest, compiled_batch_size=largest))
        start += largest

    remaining = item_count - start
    if remaining:
        bucket = next(size for size in normalized if size >= remaining)
        plans.append(BatchPlan(start=start, stop=item_count, compiled_batch_size=bucket))
    return tuple(plans)
