"""Exact tiled enumeration of the two-gene genotype-to-pattern map."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict
from typing import Any

import numpy as np

from .runtime import Engine
from .spatial import (
    _requires_extended_inputs,
    _trajectories_from_ca,
    ca_from_descriptors,
    pair_promoter_matrix,
)
from .specs import ExecutionSpec, SpatialSpec, digest
from .tables import Tables


def _descriptor_key(code: int) -> str:
    return "".join(str((code >> (2 * index)) & 3) for index in range(6)) + str(code >> 12)


def _pattern_id(trajectory: np.ndarray, protocol: str) -> str:
    content = np.asarray(trajectory, dtype="<i4", order="C").tobytes()
    return hashlib.sha256(b"toylife2-pattern-v1\0" + protocol.encode() + content).hexdigest()


def _spatial_protocol(config: SpatialSpec) -> str:
    data = asdict(config)
    data.pop("target")  # A target is scored against a trajectory; it does not generate one.
    return digest(data)


def _merge_histogram(into: dict[str, int], source: Mapping[str, int]) -> None:
    for key, amount in source.items():
        into[key] = into.get(key, 0) + int(amount)


def merge_map_shards(shards: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge complete adjacent pair intervals with exact integer checksums."""
    ordered = sorted(shards, key=lambda row: int(row["ordered_pair_start"]))
    if not ordered:
        raise ValueError("at least one completed map shard is required")
    first = ordered[0]
    result: dict[str, Any] = {
        "ordered_pair_start": first["ordered_pair_start"],
        "ordered_pair_stop": ordered[-1]["ordered_pair_stop"],
        "next_pair": ordered[-1]["ordered_pair_stop"],
        "total_ordered_pairs": first["total_ordered_pairs"],
        "total_genotypes": 0,
        "diffusion": first["diffusion"],
        "spatial_digest": first["spatial_digest"],
        "table_digest": first["table_digest"],
        "status": "complete",
        "ca_to_pattern": {},
    }
    fields = ("logic_histogram", "grn_histogram", "ca_histogram", "pattern_histogram")
    for field in fields:
        result[field] = {}
    cursor = int(first["ordered_pair_start"])
    for shard in ordered:
        if shard["status"] != "complete" or int(shard["ordered_pair_start"]) != cursor:
            raise ValueError("map shards must be complete and adjacent")
        for key in ("total_ordered_pairs", "diffusion", "spatial_digest", "table_digest"):
            if shard[key] != result[key]:
                raise ValueError(f"incompatible map shard: {key}")
        cursor = int(shard["ordered_pair_stop"])
        result["total_genotypes"] = int(result["total_genotypes"]) + int(shard["total_genotypes"])
        for field in fields:
            _merge_histogram(result[field], shard[field])
        for rule, pattern in shard["ca_to_pattern"].items():
            prior = result["ca_to_pattern"].get(rule)
            if prior is not None and prior != pattern:
                raise ValueError("map shards disagree on a rule's tissue pattern")
            result["ca_to_pattern"][rule] = pattern
    for field in fields:
        if sum(result[field].values()) != result["total_genotypes"]:
            raise AssertionError(f"merged {field} does not conserve genotype multiplicity")
        result[field] = dict(sorted(result[field].items()))
    if (
        result["ordered_pair_start"] == 0
        and result["ordered_pair_stop"] == result["total_ordered_pairs"]
        and result["total_genotypes"] != 1 << 40
    ):
        raise AssertionError("complete map must contain exactly 2**40 genotypes")
    result["ca_to_pattern"] = dict(sorted(result["ca_to_pattern"].items()))
    result["pattern_rules"] = _pattern_rules(result["ca_to_pattern"])
    return result


def _pattern_rules(ca_to_pattern: Mapping[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for rule, pattern in ca_to_pattern.items():
        grouped[pattern].append(rule)
    return {key: sorted(values) for key, values in sorted(grouped.items())}


def enumerate_map(
    tables: Tables,
    execution: ExecutionSpec | None = None,
    *,
    spatial: SpatialSpec | None = None,
    diffusion: str | None = None,
    start_pair: int = 0,
    stop_pair: int | None = None,
    tile_pairs: int = 64,
    max_tiles: int | None = None,
    checkpoint_interval_tiles: int = 1,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    resume_checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Count logic, GRNs, local rules, and tissue patterns over pair shards.

    Ordered protein pairs and all sixteen promoters per gene cover 2**40
    genotypes. The caller can persist a tile checkpoint and resume exactly.
    """
    config = spatial or SpatialSpec(diffusion=diffusion or "A")
    mode = config.diffusion
    if mode not in ("none", "A", "B", "both", "all"):
        raise ValueError("unknown diffusion mode")
    if diffusion is not None and diffusion != mode:
        raise ValueError("diffusion must agree with spatial.diffusion")
    if type(tile_pairs) is not int or tile_pairs < 1:
        raise ValueError("tile_pairs must be positive")
    if max_tiles is not None and (type(max_tiles) is not int or max_tiles < 1):
        raise ValueError("max_tiles must be positive")
    if type(checkpoint_interval_tiles) is not int or checkpoint_interval_tiles < 1:
        raise ValueError("checkpoint_interval_tiles must be positive")
    count = np.bincount(np.asarray(tables.gene_to_protein, np.int32), minlength=2711)
    proteins = len(count)
    total_pairs = proteins * proteins
    end = total_pairs if stop_pair is None else stop_pair
    if not 0 <= start_pair <= end <= total_pairs:
        raise ValueError("invalid ordered protein-pair interval")
    protocol = _spatial_protocol(config)
    identity = digest(
        {
            "table_digest": str(tables.digest),
            "spatial_digest": protocol,
            "diffusion": mode,
            "start_pair": start_pair,
            "stop_pair": end,
        }
    )
    histograms: dict[str, dict[str, int]] = {
        "logic_histogram": defaultdict(int),
        "grn_histogram": defaultdict(int),
        "ca_histogram": defaultdict(int),
        "pattern_histogram": defaultdict(int),
    }
    ca_to_pattern: dict[str, str] = {}
    cursor = start_pair
    total_weight = 0
    if resume_checkpoint is not None:
        if resume_checkpoint.get("identity") != identity:
            raise ValueError("map checkpoint has a different table, protocol, or pair interval")
        cursor = int(resume_checkpoint["next_pair"])
        if cursor < start_pair or cursor > end:
            raise ValueError("map checkpoint cursor is outside the requested pair interval")
        total_weight = int(resume_checkpoint["total_genotypes"])
        for field in histograms:
            _merge_histogram(histograms[field], resume_checkpoint[field])
        ca_to_pattern = dict(resume_checkpoint["ca_to_pattern"])
    work = Engine(tables, execution)
    tiles = 0
    for first in range(cursor, end, tile_pairs):
        ids = np.arange(first, min(first + tile_pairs, end), dtype=np.int64)
        # Bounded host working set for promoter responses, descriptors, and CA rows.
        tile_bytes = len(ids) * (16 * 6 * 4 + 256 * (7 + 1 + 64) * 4)
        if tile_bytes > work.execution.memory_budget_mib * 1024 * 1024:
            raise MemoryError("one weighted-map tile exceeds the execution memory budget")
        left, right = ids // proteins, ids % proteins
        weights = count[left].astype(np.int64) * count[right].astype(np.int64)
        responses, present = pair_promoter_matrix(
            tables,
            np.column_stack((left, right)),
            engine=work,
        )
        descriptors = np.empty((len(ids), 16, 16, 7), dtype=np.int32)
        descriptors[..., :6] = (responses[:, :, None, :] << 1) | responses[:, None, :, :]
        descriptors[..., :6] &= np.where(left[:, None, None, None] > 0, 3, 1)
        descriptors[..., :6] &= np.where(right[:, None, None, None] > 0, 3, 2)
        descriptors[..., 6] = present[:, None, None]
        flat = descriptors.reshape(-1, 7)
        row_weights = np.broadcast_to(weights[:, None, None], (len(ids), 16, 16)).ravel()
        codes = np.zeros(len(flat), dtype=np.int32)
        for index in range(6):
            codes |= flat[:, index] << (2 * index)
        codes |= flat[:, 6] << 12
        # Weighted bincount uses float64. Every tile sums at most the 2**40
        # complete map, below float64's exact-integer limit of 2**53.
        counts = np.bincount(codes, weights=row_weights, minlength=8192)
        active = np.flatnonzero(counts)
        distinct = np.empty((len(active), 7), dtype=np.int32)
        for index in range(6):
            distinct[:, index] = (active >> (2 * index)) & 3
        distinct[:, 6] = active >> 12
        ca = ca_from_descriptors(distinct, mode)
        ca_keys = ["".join(map(str, row)) for row in ca]
        if _requires_extended_inputs(config):
            pattern_ca = ca_from_descriptors(distinct, mode, input_states=6)
        else:
            pattern_ca = ca
        pattern_keys = [
            f"{int(flag)}:" + "".join(map(str, row))
            for flag, row in zip(distinct[:, 6], pattern_ca, strict=True)
        ]
        new_ca: dict[str, tuple[np.ndarray, int]] = {}
        for key, row, flag in zip(pattern_keys, pattern_ca, distinct[:, 6], strict=True):
            if key not in ca_to_pattern:
                new_ca[key] = (row, int(flag))
        if new_ca:
            keys = list(new_ca)
            trajectories = _trajectories_from_ca(
                np.stack([new_ca[key][0] for key in keys]),
                config,
                work,
                np.asarray([new_ca[key][1] for key in keys], dtype=np.int32),
            )
            for key, trajectory in zip(keys, trajectories, strict=True):
                ca_to_pattern[key] = _pattern_id(trajectory, protocol)
        for code, ca_key, pattern_key in zip(active, ca_keys, pattern_keys, strict=True):
            amount = int(counts[code])
            grn = _descriptor_key(int(code))
            histograms["logic_histogram"][grn[:6]] += amount
            histograms["grn_histogram"][grn] += amount
            histograms["ca_histogram"][ca_key] += amount
            histograms["pattern_histogram"][ca_to_pattern[pattern_key]] += amount
        total_weight += int(weights.sum()) * 256
        cursor = int(ids[-1]) + 1
        tiles += 1
        if checkpoint_callback is not None and (
            tiles % checkpoint_interval_tiles == 0
            or cursor == end
            or (max_tiles is not None and tiles >= max_tiles)
        ):
            checkpoint_callback(
                {
                    "schema_version": 1,
                    "identity": identity,
                    "next_pair": cursor,
                    "total_genotypes": total_weight,
                    "ca_to_pattern": dict(sorted(ca_to_pattern.items())),
                    **{field: dict(sorted(values.items())) for field, values in histograms.items()},
                }
            )
        if max_tiles is not None and tiles >= max_tiles:
            break
    for field, values in histograms.items():
        if sum(values.values()) != total_weight:
            raise AssertionError(f"{field} does not conserve genotype multiplicity")
    if start_pair == 0 and cursor == total_pairs and total_weight != 1 << 40:
        raise AssertionError("complete two-gene map must contain exactly 2**40 genotypes")
    return {
        "ordered_pair_start": start_pair,
        "ordered_pair_stop": end,
        "next_pair": cursor,
        "total_ordered_pairs": total_pairs,
        "total_genotypes": total_weight,
        "diffusion": mode,
        "spatial_digest": protocol,
        "table_digest": str(tables.digest),
        "status": "complete" if cursor == end else "partial",
        "ca_to_pattern": dict(sorted(ca_to_pattern.items())),
        "pattern_rules": _pattern_rules(ca_to_pattern),
        **{field: dict(sorted(values.items())) for field, values in histograms.items()},
    }
