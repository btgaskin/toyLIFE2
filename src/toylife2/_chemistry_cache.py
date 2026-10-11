"""Engine-owned, byte-bounded immutable host chemistry plans."""

from __future__ import annotations

import sys
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np


def _bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, Mapping):
        if isinstance(value, MappingProxyType):
            size += sys.getsizeof(dict(value))
        size += sum(_bytes(k, seen) + _bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (tuple, list)):
        size += sum(_bytes(v, seen) for v in value)
    elif hasattr(value, "__dict__") and not isinstance(value, np.ndarray):
        size += _bytes(vars(value), seen)
    return size


class GraphPlan:
    def __init__(self, graph, compiled):
        self.graph = graph
        self.compiled = compiled
        self.bytes = _bytes((graph, compiled)) + 512


class GraphCache:
    """Active plans stay pinned by their batches; idle plans use bounded LRU."""

    def __init__(self, budget):
        self.limit = min(32 * 1024 * 1024, budget // 16)
        self.live = weakref.WeakValueDictionary()
        self.idle = OrderedDict()
        self.hits = self.misses = 0
        self._live_bytes = 0
        self._sizes = {}

    @property
    def bytes(self):
        return (
            self._live_bytes
            + sys.getsizeof(self.idle)
            + sys.getsizeof(vars(self.live).get("data", {}))
            + sys.getsizeof(self._sizes)
            + 512
        )

    def reclaim(self, available):
        while self.idle and self.bytes > min(self.limit, max(0, available)):
            self.idle.popitem(last=False)

    def get(self, key):
        plan = self.live.get(key)
        if plan is None:
            self.misses += 1
        else:
            self.hits += 1
            self.idle[key] = plan
            self.idle.move_to_end(key)
        return plan

    def put(self, key, graph, compiled):
        plan = GraphPlan(graph, compiled)
        size = plan.bytes + _bytes(key) + 512
        self._sizes[key] = size
        self._live_bytes += size
        weakref.finalize(plan, type(self)._released, weakref.ref(self), key)
        self.live[key] = plan
        self.idle[key] = plan
        self.reclaim(self.limit)
        return plan

    @staticmethod
    def _released(cache_ref, key):
        cache = cache_ref()
        if cache is not None:
            cache._live_bytes -= cache._sizes.pop(key)
