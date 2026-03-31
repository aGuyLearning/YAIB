"""Union holdout across many (task, dataset) corpora for merged SSL pretrain.

For each pair, compute holdout stay IDs with the same logic as corpus_split, then take
the set union and filter the merged corpus to stays not in that union.
"""

from __future__ import annotations

import json
import logging
import zlib
from collections import defaultdict
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
from icu_benchmarks.data.pooled_stay_id import apply_pooled_stay_id_suffix

logger = logging.getLogger(__name__)

# Task folder names (data_root/<task>/...) with continuous regression outcomes.
# Classification-style label stratification (sklearn) is inappropriate and differs from
# YAIB regression CV (no stratify on label). Align holdout with that behavior.
REGRESSION_HOLDOUT_TASK_SLUGS: frozenset[str] = frozenset({"los", "kidney_function"})


def effective_holdout_stratify(stratify_requested: bool, task: str) -> bool:
    """Whether to stratify the holdout split for this task (per-pair)."""
    if not stratify_requested:
        return False
    if task in REGRESSION_HOLDOUT_TASK_SLUGS:
        return False
    return True


def _manifest_regression_stratify_fields(
    stratify_requested: bool, corpora: Sequence[TaskDatasetCorpus]
) -> dict[str, Any]:
    if not stratify_requested:
        return {}
    reg_present = sorted({c.task for c in corpora if c.task in REGRESSION_HOLDOUT_TASK_SLUGS})
    if not reg_present:
        return {}
    return {
        "holdout_stratify_regression_tasks_unstratified": reg_present,
        "holdout_stratify_note": (
            "stratify=true uses label stratification only for classification tasks; "
            "los and kidney_function use unstratified holdout (regression), matching YAIB regression CV."
        ),
    }


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


def _compute_per_pair_holdouts(
    corpora: Sequence[TaskDatasetCorpus],
    *,
    holdout_fraction: float,
    base_seed: int,
    group_col: str,
    label_col: str,
    stratify: bool,
    balance_by_pool_source: bool,
    outcome_basename: str = "outc.parquet",
    pooled_index_for_dataset: dict[str, int] | None = None,
) -> dict[str, set[int]]:
    """Holdout stay sets keyed by task::dataset (pair_key).

    If ``pooled_index_for_dataset`` is set, each pair's ``dataset`` slug must appear as a key;
    holdout IDs are remapped with ``apply_pooled_stay_id_suffix`` to match merged pooled parquets.
    """
    per_pair: dict[str, set[int]] = {}
    for c in corpora:
        outc_path = Path(c.path) / outcome_basename
        if not outc_path.is_file():
            raise FileNotFoundError(f"Outcome parquet not found for {c.pair_key()}: {outc_path}")
        outcome = pl.read_parquet(outc_path)
        if group_col not in outcome.columns:
            raise ValueError(f"{c.pair_key()}: group_col {group_col!r} missing from outcome")

        seed = pair_seed(c.task, c.dataset, base_seed)
        pair_stratify = effective_holdout_stratify(stratify, c.task)
        if stratify and not pair_stratify:
            logger.info(
                "Holdout for %s: stratify disabled (regression task; random split, like YAIB regression CV).",
                c.pair_key(),
            )
        _pre, holdout = split_stays_pretrain_holdout(
            outcome,
            group_col,
            label_col,
            holdout_fraction,
            seed,
            stratify=pair_stratify,
            balance_by_pool_source=balance_by_pool_source,
            split_context=c.pair_key(),
        )
        if pooled_index_for_dataset is not None:
            if c.dataset not in pooled_index_for_dataset:
                raise ValueError(
                    f"{c.pair_key()}: dataset slug {c.dataset!r} not in pooled merge-order map "
                    f"(keys: {sorted(pooled_index_for_dataset.keys())})"
                )
            idx = pooled_index_for_dataset[c.dataset]
            holdout = {apply_pooled_stay_id_suffix(sid, idx) for sid in holdout}
        per_pair[c.pair_key()] = holdout
    return per_pair


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
    pooled_index_for_dataset: dict[str, int] | None = None,
) -> tuple[set[int], dict[str, set[int]]]:
    """Return (union_holdout_stays, per_pair_holdout keyed by pair_key())."""
    per_pair = _compute_per_pair_holdouts(
        corpora,
        holdout_fraction=holdout_fraction,
        base_seed=base_seed,
        group_col=group_col,
        label_col=label_col,
        stratify=stratify,
        balance_by_pool_source=balance_by_pool_source,
        outcome_basename=outcome_basename,
        pooled_index_for_dataset=pooled_index_for_dataset,
    )
    union: set[int] = set()
    for s in per_pair.values():
        union |= s
    return union, per_pair


@dataclass(frozen=True)
class HierarchicalUnionResult:
    """task×site holdouts, unions per dataset (site), and corpus-level union."""

    corpus_union: set[int]
    per_pair: dict[str, set[int]]
    per_dataset: dict[str, set[int]]


def discover_single_site_corpora(
    data_root: Path,
    tasks: Sequence[str],
    datasets: Sequence[str],
    *,
    require_all_pairs: bool = False,
    outcome_basename: str = "outc.parquet",
) -> list[TaskDatasetCorpus]:
    """Discover ``data_root/<task>/<dataset>/outc.parquet`` for single-site YAIB layouts."""
    data_root = Path(data_root).resolve()
    found: list[TaskDatasetCorpus] = []
    missing: list[str] = []
    for task in tasks:
        task = task.strip()
        if not task:
            continue
        for dataset in datasets:
            dataset = dataset.strip()
            if not dataset:
                continue
            path = data_root / task / dataset
            outc = path / outcome_basename
            if outc.is_file():
                found.append(TaskDatasetCorpus(task=task, dataset=dataset, path=path))
            else:
                missing.append(f"{task}/{dataset}")
    if require_all_pairs and missing:
        preview = ", ".join(missing[:12])
        more = f" (+{len(missing) - 12} more)" if len(missing) > 12 else ""
        raise FileNotFoundError(
            f"require_all_pairs: missing {outcome_basename} for: {preview}{more}"
        )
    if not found:
        raise FileNotFoundError(
            f"No corpora found under {data_root} for given tasks×datasets (expected {outcome_basename})"
        )
    return found


def compute_hierarchical_union_holdout(
    corpora: Sequence[TaskDatasetCorpus],
    *,
    holdout_fraction: float,
    base_seed: int,
    group_col: str,
    label_col: str,
    stratify: bool,
    balance_by_pool_source: bool,
    outcome_basename: str = "outc.parquet",
    pooled_index_for_dataset: dict[str, int] | None = None,
) -> HierarchicalUnionResult:
    """Union holdouts per task×site, then per dataset (site), then full corpus.

    Corpus-level union equals the flat union over all pair holdouts.
    """
    per_pair = _compute_per_pair_holdouts(
        corpora,
        holdout_fraction=holdout_fraction,
        base_seed=base_seed,
        group_col=group_col,
        label_col=label_col,
        stratify=stratify,
        balance_by_pool_source=balance_by_pool_source,
        outcome_basename=outcome_basename,
        pooled_index_for_dataset=pooled_index_for_dataset,
    )
    per_dataset: dict[str, set[int]] = defaultdict(set)
    for c in corpora:
        per_dataset[c.dataset] |= per_pair[c.pair_key()]
    corpus_union: set[int] = set()
    for s in per_dataset.values():
        corpus_union |= s
    flat_union = set()
    for s in per_pair.values():
        flat_union |= s
    if corpus_union != flat_union:
        raise RuntimeError("internal error: hierarchical corpus_union != flat union over pairs")
    return HierarchicalUnionResult(
        corpus_union=corpus_union,
        per_pair=dict(per_pair),
        per_dataset={k: set(v) for k, v in sorted(per_dataset.items())},
    )


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
    pooled_stay_id_suffix_applied: bool = False,
    pooled_merge_order: list[str] | None = None,
) -> dict[str, Any]:
    per_pair_json: dict[str, Any] = {}
    for key, hset in sorted(per_pair_holdout.items()):
        per_pair_json[key] = {
            "holdout_stay_ids": sorted(hset),
            "n_holdout_stays": len(hset),
        }
    out: dict[str, Any] = {
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
    if pooled_stay_id_suffix_applied:
        out["pooled_stay_id_suffix_applied"] = True
        out["note_holdout_ids"] = "stay_id values match merged pooled parquets (suffix applied)"
    if pooled_merge_order:
        out["pooled_merge_order"] = list(pooled_merge_order)
    return out


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


def write_merged_holdout_subset(
    merged_input_dir: Path,
    output_holdout_dir: Path,
    holdout_stays: set[int],
    *,
    group_col: str,
    parquet_names: tuple[str, ...] = DEFAULT_PARQUETS,
    outcome_basename: str = "outc.parquet",
) -> dict[str, int]:
    """Write parquets containing only rows whose ``group_col`` is in ``holdout_stays`` ∩ merged outcome stays."""
    merged_input_dir = Path(merged_input_dir).resolve()
    outc_path = merged_input_dir / outcome_basename
    if not outc_path.is_file():
        raise FileNotFoundError(f"Outcome parquet not found: {outc_path}")

    outcome = pl.read_parquet(outc_path)
    if group_col not in outcome.columns:
        raise ValueError(f"group_col {group_col!r} missing from merged outcome")

    all_stays = {int(x) for x in outcome[group_col].unique().to_list()}
    stays_to_write = holdout_stays & all_stays
    if not stays_to_write:
        logger.warning("No holdout stays intersect merged outcome; holdout parquets may be empty.")

    files = discover_parquet_files(merged_input_dir, parquet_names)
    if not files:
        raise FileNotFoundError(f"No parquet files from {parquet_names} found in {merged_input_dir}")

    for base, path in files:
        if base != outcome_basename:
            validate_segment_stays_subset(path, group_col, all_stays, base)

    output_holdout_dir = Path(output_holdout_dir).resolve()
    output_holdout_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for base, path in files:
        counts[base] = filter_parquet_to_stays(path, group_col, stays_to_write, output_holdout_dir / base)

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
    pooled_index_for_dataset: dict[str, int] | None = None,
    pooled_merge_order: list[str] | None = None,
    output_holdout_dir: Path | None = None,
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
        pooled_index_for_dataset=pooled_index_for_dataset,
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

    if output_holdout_dir is not None:
        write_merged_holdout_subset(
            merged_input_dir,
            output_holdout_dir,
            union_h,
            group_col=group_col,
            parquet_names=parquet_names,
            outcome_basename=outcome_basename,
        )

    suffix_on = pooled_index_for_dataset is not None
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
        pooled_stay_id_suffix_applied=suffix_on,
        pooled_merge_order=pooled_merge_order,
    )
    manifest.update(_manifest_regression_stratify_fields(stratify, corpora))

    manifest_path = Path(manifest_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    if output_holdout_dir is not None:
        hod = Path(output_holdout_dir).resolve()
        hod.mkdir(parents=True, exist_ok=True)
        with open(hod / manifest_path.name, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

    return manifest


def build_hierarchical_manifest(
    *,
    holdout_fraction: float,
    base_seed: int,
    group_col: str,
    label_col: str,
    stratify: bool,
    balance_by_pool_source: bool,
    outcome_basename: str,
    hierarchical: HierarchicalUnionResult,
    merged_stays: set[int],
    extra_union_not_in_merged: set[int],
    data_root: str | None = None,
    tasks_scanned: list[str] | None = None,
    datasets_scanned: list[str] | None = None,
    pooled_stay_id_suffix_applied: bool = False,
    pooled_merge_order: list[str] | None = None,
) -> dict[str, Any]:
    per_pair_json: dict[str, Any] = {}
    for key, hset in sorted(hierarchical.per_pair.items()):
        per_pair_json[key] = {
            "holdout_stay_ids": sorted(hset),
            "n_holdout_stays": len(hset),
        }
    per_dataset_json: dict[str, Any] = {}
    for ds, hset in sorted(hierarchical.per_dataset.items()):
        task_keys = sorted(k for k in hierarchical.per_pair if k.endswith(f"::{ds}"))
        per_dataset_json[ds] = {
            "task_pair_keys": task_keys,
            "n_holdout_stays": len(hset),
            "union_holdout_stay_ids": sorted(hset),
        }
    out: dict[str, Any] = {
        "kind": "hierarchical_union_holdout_pretrain",
        "holdout_fraction": holdout_fraction,
        "base_seed": base_seed,
        "group_col": group_col,
        "label_col": label_col,
        "stratify": stratify,
        "balance_by_pool_source": balance_by_pool_source,
        "outcome_basename": outcome_basename,
        "n_merged_stays": len(merged_stays),
        "n_union_holdout_stays": len(hierarchical.corpus_union),
        "n_union_holdout_in_merged": len(hierarchical.corpus_union & merged_stays),
        "n_extra_union_not_in_merged": len(extra_union_not_in_merged),
        "union_holdout_stay_ids": sorted(hierarchical.corpus_union),
        "per_pair": per_pair_json,
        "per_dataset": per_dataset_json,
    }
    if data_root is not None:
        out["data_root"] = data_root
    if tasks_scanned is not None:
        out["tasks_scanned"] = list(tasks_scanned)
    if datasets_scanned is not None:
        out["datasets_scanned"] = list(datasets_scanned)
    if pooled_stay_id_suffix_applied:
        out["pooled_stay_id_suffix_applied"] = True
        out["note_holdout_ids"] = "stay_id values match merged pooled parquets (suffix applied)"
    if pooled_merge_order:
        out["pooled_merge_order"] = list(pooled_merge_order)
    return out


def run_hierarchical_union_holdout_pretrain(
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
    data_root: Path | None = None,
    tasks_scanned: list[str] | None = None,
    datasets_scanned: list[str] | None = None,
    pooled_index_for_dataset: dict[str, int] | None = None,
    pooled_merge_order: list[str] | None = None,
    output_holdout_dir: Path | None = None,
) -> dict[str, Any]:
    """Hierarchical union holdout; write pretrain-only merged dir + nested manifest."""
    hier = compute_hierarchical_union_holdout(
        corpora,
        holdout_fraction=holdout_fraction,
        base_seed=base_seed,
        group_col=group_col,
        label_col=label_col,
        stratify=stratify,
        balance_by_pool_source=balance_by_pool_source,
        outcome_basename=outcome_basename,
        pooled_index_for_dataset=pooled_index_for_dataset,
    )
    merged_stays = merged_outcome_stays(merged_input_dir, group_col, outcome_basename)
    extra = validate_union_against_merged(hier.corpus_union, merged_stays, strict=strict)

    write_pretrain_excluding_stays(
        merged_input_dir,
        output_pretrain_dir,
        hier.corpus_union,
        group_col=group_col,
        parquet_names=parquet_names,
        outcome_basename=outcome_basename,
    )

    if output_holdout_dir is not None:
        write_merged_holdout_subset(
            merged_input_dir,
            output_holdout_dir,
            hier.corpus_union,
            group_col=group_col,
            parquet_names=parquet_names,
            outcome_basename=outcome_basename,
        )

    suffix_on = pooled_index_for_dataset is not None
    manifest = build_hierarchical_manifest(
        holdout_fraction=holdout_fraction,
        base_seed=base_seed,
        group_col=group_col,
        label_col=label_col,
        stratify=stratify,
        balance_by_pool_source=balance_by_pool_source,
        outcome_basename=outcome_basename,
        hierarchical=hier,
        merged_stays=merged_stays,
        extra_union_not_in_merged=extra,
        data_root=str(data_root.resolve()) if data_root is not None else None,
        tasks_scanned=tasks_scanned,
        datasets_scanned=datasets_scanned,
        pooled_stay_id_suffix_applied=suffix_on,
        pooled_merge_order=pooled_merge_order,
    )
    manifest.update(_manifest_regression_stratify_fields(stratify, corpora))

    manifest_path = Path(manifest_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    if output_holdout_dir is not None:
        hod = Path(output_holdout_dir).resolve()
        hod.mkdir(parents=True, exist_ok=True)
        with open(hod / manifest_path.name, "w", encoding="utf-8") as f:
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
