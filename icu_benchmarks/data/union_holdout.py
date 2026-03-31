"""Union holdout across many (task, dataset) corpora for merged SSL pretrain.

For each pair, compute holdout stay IDs with the same logic as corpus_split, then take
the set union and filter the merged corpus to stays not in that union.
"""

from __future__ import annotations

import json
import logging
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from icu_benchmarks.data.corpus_split import (
    DEFAULT_PARQUETS,
    discover_parquet_files,
    filter_parquet_to_stays,
    split_stays_pretrain_holdout,
    validate_segment_stays_subset,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskDatasetCorpus:
    """One supervised (task, dataset) layout with its own outcome table."""

    task: str
    dataset: str
    path: Path  # directory containing outcome parquet

    def pair_key(self) -> str:
        return f"{self.task}::{self.dataset}"


def pair_seed(task: str, dataset: str, base_seed: int) -> int:
    """Reproducible per-pair offset so splits differ across pairs."""
    h = zlib.crc32(f"{task}:{dataset}".encode()) & 0xFFFFFFFF
    return int((base_seed + h) % (2**31))


def compute_union_holdout(
    corpora: Sequence[TaskDatasetCorpus],
    *,
    holdout_fraction: float,
    base_seed: int,
    group_col: str,
    label_col: str,
    stratify: bool,
    balance_by_pool_source: bool,
    outcome_basename: str = "outc.parquet",
) -> tuple[set[int], dict[str, set[int]]]:
    """Return (union_holdout_stays, per_pair_holdout keyed by pair_key())."""
    per_pair: dict[str, set[int]] = {}
    union: set[int] = set()

    for c in corpora:
        outc_path = Path(c.path) / outcome_basename
        if not outc_path.is_file():
            raise FileNotFoundError(f"Outcome parquet not found for {c.pair_key()}: {outc_path}")
        outcome = pl.read_parquet(outc_path)
        if group_col not in outcome.columns:
            raise ValueError(f"{c.pair_key()}: group_col {group_col!r} missing from outcome")

        seed = pair_seed(c.task, c.dataset, base_seed)
        _pre, holdout = split_stays_pretrain_holdout(
            outcome,
            group_col,
            label_col,
            holdout_fraction,
            seed,
            stratify=stratify,
            balance_by_pool_source=balance_by_pool_source,
        )
        key = c.pair_key()
        per_pair[key] = holdout
        union |= holdout

    return union, per_pair


def merged_outcome_stays(
    merged_input_dir: Path,
    group_col: str,
    outcome_basename: str = "outc.parquet",
) -> set[int]:
    path = Path(merged_input_dir) / outcome_basename
    if not path.is_file():
        raise FileNotFoundError(f"Merged outcome parquet not found: {path}")
    df = pl.read_parquet(path, columns=[group_col])
    return {int(x) for x in df[group_col].unique().to_list()}


def validate_union_against_merged(
    union_holdout: set[int],
    merged_stays: set[int],
    *,
    strict: bool,
) -> set[int]:
    """Return stay IDs in the union that are absent from merged outcome (ID mismatch)."""
    extra = union_holdout - merged_stays
    if not extra:
        return extra
    msg = (
        f"{len(extra)} holdout stay id(s) from pair corpora are not present in merged outcome, "
        f"e.g. {sorted(extra)[:8]}"
    )
    if strict:
        raise ValueError(msg)
    logger.warning(msg)
    return extra


def build_union_manifest(
    *,
    holdout_fraction: float,
    base_seed: int,
    group_col: str,
    label_col: str,
    stratify: bool,
    balance_by_pool_source: bool,
    outcome_basename: str,
    union_holdout: set[int],
    per_pair_holdout: dict[str, set[int]],
    merged_stays: set[int],
    extra_union_not_in_merged: set[int],
) -> dict[str, Any]:
    per_pair_json: dict[str, Any] = {}
    for key, hset in sorted(per_pair_holdout.items()):
        per_pair_json[key] = {
            "holdout_stay_ids": sorted(hset),
            "n_holdout_stays": len(hset),
        }
    return {
        "kind": "union_holdout_pretrain",
        "holdout_fraction": holdout_fraction,
        "base_seed": base_seed,
        "group_col": group_col,
        "label_col": label_col,
        "stratify": stratify,
        "balance_by_pool_source": balance_by_pool_source,
        "outcome_basename": outcome_basename,
        "n_merged_stays": len(merged_stays),
        "n_union_holdout_stays": len(union_holdout),
        "n_union_holdout_in_merged": len(union_holdout & merged_stays),
        "n_extra_union_not_in_merged": len(extra_union_not_in_merged),
        "union_holdout_stay_ids": sorted(union_holdout),
        "per_pair": per_pair_json,
    }


def write_pretrain_excluding_stays(
    merged_input_dir: Path,
    output_pretrain_dir: Path,
    excluded_stays: set[int],
    *,
    group_col: str,
    parquet_names: tuple[str, ...] = DEFAULT_PARQUETS,
    outcome_basename: str = "outc.parquet",
) -> dict[str, int]:
    """Write filtered parquets containing only stays with stay_id not in excluded_stays.

    Returns map basename -> row count written.
    """
    merged_input_dir = Path(merged_input_dir).resolve()
    outc_path = merged_input_dir / outcome_basename
    if not outc_path.is_file():
        raise FileNotFoundError(f"Outcome parquet not found: {outc_path}")

    outcome = pl.read_parquet(outc_path)
    if group_col not in outcome.columns:
        raise ValueError(f"group_col {group_col!r} missing from merged outcome")

    all_stays = {int(x) for x in outcome[group_col].unique().to_list()}
    pretrain_stays = all_stays - excluded_stays

    files = discover_parquet_files(merged_input_dir, parquet_names)
    if not files:
        raise FileNotFoundError(f"No parquet files from {parquet_names} found in {merged_input_dir}")

    for base, path in files:
        if base != outcome_basename:
            validate_segment_stays_subset(path, group_col, all_stays, base)

    output_pretrain_dir = Path(output_pretrain_dir).resolve()
    output_pretrain_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for base, path in files:
        counts[base] = filter_parquet_to_stays(path, group_col, pretrain_stays, output_pretrain_dir / base)

    return counts


def run_union_holdout_pretrain(
    corpora: Sequence[TaskDatasetCorpus],
    merged_input_dir: Path,
    output_pretrain_dir: Path,
    manifest_path: Path,
    *,
    holdout_fraction: float,
    base_seed: int,
    group_col: str,
    label_col: str,
    stratify: bool,
    balance_by_pool_source: bool,
    parquet_names: tuple[str, ...] = DEFAULT_PARQUETS,
    outcome_basename: str = "outc.parquet",
    strict: bool = False,
) -> dict[str, Any]:
    """Compute union holdout from pair corpora; write pretrain-only merged dir + manifest."""
    union_h, per_pair = compute_union_holdout(
        corpora,
        holdout_fraction=holdout_fraction,
        base_seed=base_seed,
        group_col=group_col,
        label_col=label_col,
        stratify=stratify,
        balance_by_pool_source=balance_by_pool_source,
        outcome_basename=outcome_basename,
    )
    merged_stays = merged_outcome_stays(merged_input_dir, group_col, outcome_basename)
    extra = validate_union_against_merged(union_h, merged_stays, strict=strict)

    write_pretrain_excluding_stays(
        merged_input_dir,
        output_pretrain_dir,
        union_h,
        group_col=group_col,
        parquet_names=parquet_names,
        outcome_basename=outcome_basename,
    )

    manifest = build_union_manifest(
        holdout_fraction=holdout_fraction,
        base_seed=base_seed,
        group_col=group_col,
        label_col=label_col,
        stratify=stratify,
        balance_by_pool_source=balance_by_pool_source,
        outcome_basename=outcome_basename,
        union_holdout=union_h,
        per_pair_holdout=per_pair,
        merged_stays=merged_stays,
        extra_union_not_in_merged=extra,
    )
    manifest_path = Path(manifest_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest


def parse_pair_arg(spec: str) -> TaskDatasetCorpus:
    """Parse ``task=DATASET=PATH`` (PATH may contain ``=`` if using JSON list instead)."""
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --pair {spec!r}: expected exactly two '=' separators as task=DATASET=PATH"
        )
    task, dataset, path_s = parts[0].strip(), parts[1].strip(), parts[2].strip()
    if not task or not dataset or not path_s:
        raise ValueError(f"Invalid --pair {spec!r}: empty task, dataset, or path")
    return TaskDatasetCorpus(task=task, dataset=dataset, path=Path(path_s))


def corpora_from_json(path: Path) -> list[TaskDatasetCorpus]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("pairs JSON must be a list of objects with task, dataset, path")
    out: list[TaskDatasetCorpus] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"pairs JSON entry {i} must be an object")
        try:
            out.append(
                TaskDatasetCorpus(
                    task=str(item["task"]),
                    dataset=str(item["dataset"]),
                    path=Path(str(item["path"])),
                )
            )
        except KeyError as e:
            raise ValueError(f"pairs JSON entry {i} missing key {e}") from e
    return out
