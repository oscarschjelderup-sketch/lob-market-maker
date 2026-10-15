"""A priority queue of timed events. Ties are broken by insertion order, so a run is reproducible."""

from __future__ import annotations

import heapq
from enum import IntEnum
from typing import Any


class Kind(IntEnum):
    LIMIT = 1  # a noise limit order arrives (payload: side)
    MARKET = 2  # a noise market order arrives (payload: side)
    INFORMED = 3  # an informed trader looks at the book (payload: None)
    CANCEL = 4  # a resting noise order is cancelled (payload: order id)
    MM_WAKE = 5  # the market maker looks at the book (payload: None)
    MM_DECIDE = 6  # the snapshot reaches the market maker after the latency (payload: snapshot)
    MM_ORDERS = 7  # the instructions of the market maker reach the exchange (payload: (bid, ask))
    SAMPLE = 8  # record the depth of the book, for the heatmap (payload: None)


class EventQueue:
    __slots__ = ("_heap", "_seq")

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, int, Any]] = []
        self._seq = 0

    def push(self, t: float, kind: Kind, payload: Any = None) -> None:
        heapq.heappush(self._heap, (t, self._seq, int(kind), payload))
        self._seq += 1

    def pop(self) -> tuple[float, int, Any]:
        t, _, kind, payload = heapq.heappop(self._heap)
        return t, kind, payload

    def peek_time(self) -> float | None:
        return self._heap[0][0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)
