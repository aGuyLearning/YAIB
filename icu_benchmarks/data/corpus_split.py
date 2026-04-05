"""Split a pooled YAIB corpus into disjoint pretrain vs holdout stay sets and filtered parquets."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import polars as pl
from sklearn.model_selection import train_test_split

from icu_benchmarks.data.batch_samplers import pool_source_id_from_stay_id

logger = logging.getLogger(__name__)

DEFAULT_PARQUETS = ("outc.parquet", "dyn.parquet", "sta.parquet")


def list_stays_in_order(outcome: pl.DataFrame, group_col: str) -> list[int]:
    """Stay ids in first-appearance order (aligns with typical Polars unique ordering)."""
    return [int(x) for x in outcome[group_col].unique(maintain_order=True).to_list()]


def per_stay_max_label(outcome: pl.DataFrame, group_col: str, label_col: str) -> dict[int, Any]:
    """Max label per stay (parity with stratified CV on sequence outcomes)."""
    g = outcome.group_by(group_col).agg(pl.col(label_col).max())
    rows = g.select(group_col, label_col).iter_rows(named=True)
    return {int(r[group_col]): r[label_col] for r in rows}


def _stratify_viable(labels: list) -> bool:
    counts = Counter(labels)
    return len(counts) > 0 and min(counts.values()) >= 2


def _split_stays_one_pool(
    stays: list[int],
    labels: Optional[list],
    holdout_fraction: float,
    seed: int,
    stratify: bool,
    *,
    log_context: Optional[str] = None,
) -> tuple[list[int], list[int]]:
    if not stays:
        return [], []
    if len(stays) == 1:
        return stays, []

    strat_labels = None
    if stratify and labels is not None and _stratify_viable(labels):
        strat_labels = labels
    elif stratify and labels is not None:
        ctx = f" [{log_context}]" if log_context else ""
        counts = dict(Counter(labels))
        logger.warning(
            "Stratified holdout split skipped%s: need >=2 stays per label class (min count >= 2); "
            "n_stays=%d label_counts=%s; using unstratified split.",
            ctx,
            len(stays),
            counts,
        )

    train_s, test_s = train_test_split(
        stays,
        test_size=holdout_fraction,
        random_state=seed,
        shuffle=True,
        stratify=strat_labels,
    )
    return [int(x) for x in train_s], [int(x) for x in test_s]


def split_stays_pretrain_holdout(
    outcome: pl.DataFrame,
    group_col: str,
    label_col: str,
    holdout_fraction: float,
    seed: int,
    *,
    stratify: bool,
    balance_by_pool_source: bool,
    split_context: Optional[str] = None,
) -> tuple[set[int], set[int]]:
    """Return (pretrain_stays, holdout_stays) as sets of int stay ids."""
    stays = list_stays_in_order(outcome, group_col)
    if not stays:
        return set(), set()

    label_lut = None
    labels_list: Optional[list] = None
    if label_col in outcome.columns:
        label_lut = per_stay_max_label(outcome, group_col, label_col)
        labels_list = [label_lut[s] for s in stays]
    elif stratify:
        ctx = f" [{split_context}]" if split_context else ""
        logger.warning("Label column %r not found%s; using unstratified split.", label_col, ctx)
        stratify = False

    pretrain: list[int] = []
    holdout: list[int] = []

    if balance_by_pool_source:
        by_src: dict[int, list[int]] = defaultdict(list)
        for s in stays:
            by_src[pool_source_id_from_stay_id(int(s))].append(int(s))
        for src, sid_list in sorted(by_src.items()):
            labs = [label_lut[s] for s in sid_list] if labels_list is not None else None
            bucket_ctx = (
                f"{split_context} pool_source_id={src}" if split_context else f"pool_source_id={src}"
            )
            pt, hd = _split_stays_one_pool(
                sid_list,
                labs,
                holdout_fraction,
                seed + src * 10_003,
                stratify,
                log_context=bucket_ctx,
            )
            pretrain.extend(pt)
            holdout.extend(hd)
    else:
        pt, hd = _split_stays_one_pool(
            stays,
            labels_list,
            holdout_fraction,
            seed,
            stratify,
            log_context=split_context,
        )
        pretrain.extend(pt)
        holdout.extend(hd)

    return set(pretrain), set(holdout)


def filter_parquet_to_stays(path: Path, group_col: str, stays: set[int], output_path: Path) -> int:
    """Write rows with stay_id in stays to output_path. Returns row count written."""
    df = pl.read_parquet(path)
    if group_col not in df.columns:
        raise ValueError(f"Column {group_col!r} not in {path}")
    filtered = df.filter(pl.col(group_col).is_in(list(stays)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_parquet(output_path)
    return filtered.height


def discover_parquet_files(input_dir: Path, names: tuple[str, ...]) -> list[tuple[str, Path]]:
    """Return list of (basename, path) for files that exist under input_dir."""
    if not names:
        return sorted(
            ((p.name, p) for p in input_dir.glob("*.parquet") if p.is_file()),
            key=lambda item: item[0],
        )
    found: list[tuple[str, Path]] = []
    for name in names:
        p = input_dir / name
        if p.is_file():
            found.append((name, p))
    return found


def build_manifest(
    seed: int,
    holdout_fraction: float,
    group_col: str,
    label_col: str,
    stratify: bool,
    balance_by_pool_source: bool,
    pretrain_stays: set[int],
    holdout_stays: set[int],
    outcome: pl.DataFrame,
) -> dict[str, Any]:
    pretrain_list = sorted(pretrain_stays)
    holdout_list = sorted(holdout_stays)
    manifest: dict[str, Any] = {
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "group_col": group_col,
        "label_col": label_col,
        "stratify": stratify,
        "balance_by_pool_source": balance_by_pool_source,
        "n_stays_pretrain": len(pretrain_stays),
        "n_stays_holdout": len(holdout_stays),
        "pretrain_stay_ids": pretrain_list,
        "holdout_stay_ids": holdout_list,
    }
    if label_col in outcome.columns:
        lut = per_stay_max_label(outcome, group_col, label_col)
        manifest["label_counts_pretrain"] = dict(Counter(lut[s] for s in pretrain_stays if s in lut))
        manifest["label_counts_holdout"] = dict(Counter(lut[s] for s in holdout_stays if s in lut))
    manifest["pool_source_counts_pretrain"] = {
        str(k): v for k, v in Counter(pool_source_id_from_stay_id(int(s)) for s in pretrain_stays).items()
    }
    manifest["pool_source_counts_holdout"] = {
        str(k): v for k, v in Counter(pool_source_id_from_stay_id(int(s)) for s in holdout_stays).items()
    }
    return manifest


def validate_stays_partition(
    outcome: pl.DataFrame,
    group_col: str,
    pretrain_stays: set[int],
    holdout_stays: set[int],
) -> set[int]:
    """Outcome stays must partition exactly into P and H. Returns the universe of outcome stays."""
    all_stays = {int(x) for x in outcome[group_col].unique().to_list()}
    if pretrain_stays & holdout_stays:
        raise ValueError("pretrain and holdout stay sets overlap")
    if pretrain_stays | holdout_stays != all_stays:
        raise ValueError(
            "Stays must partition outcome rows: "
            f"missing from P|H={sorted(all_stays - (pretrain_stays | holdout_stays))[:32]}, "
            f"extra in P|H={sorted((pretrain_stays | holdout_stays) - all_stays)[:32]}"
        )
    return all_stays


def validate_segment_stays_subset(
    path: Path,
    group_col: str,
    outcome_stays: set[int],
    basename: str,
) -> None:
    """Dynamic/static segments must not introduce stays absent from outcome (otherwise rows would be dropped)."""
    df = pl.read_parquet(path, columns=[group_col])
    file_stays = {int(x) for x in df[group_col].unique().to_list()}
    extra = file_stays - outcome_stays
    if extra:
        raise ValueError(f"{basename}: {len(extra)} stays not present in outcome parquet, e.g. {sorted(extra)[:5]}")


def run_corpus_split(
    input_dir: Path,
    output_pretrain_dir: Path,
    output_holdout_dir: Path,
    *,
    holdout_fraction: float,
    seed: int,
    group_col: str,
    label_col: str,
    stratify: bool,
    balance_by_pool_source: bool,
    parquet_names: tuple[str, ...] = DEFAULT_PARQUETS,
    outcome_basename: str = "outc.parquet",
) -> dict[str, Any]:
    """Split corpus; write filtered parquets and manifest JSON next to output dirs."""
    input_dir = input_dir.resolve()
    outc_path = input_dir / outcome_basename
    if not outc_path.is_file():
        raise FileNotFoundError(f"Outcome parquet not found: {outc_path}")

    outcome = pl.read_parquet(outc_path)
    if group_col not in outcome.columns:
        raise ValueError(f"group_col {group_col!r} missing from outcome")

    if stratify and label_col not in outcome.columns:
        logger.warning("Stratify requested but label column %r missing; using unstratified split.", label_col)
        stratify = False

    pretrain_stays, holdout_stays = split_stays_pretrain_holdout(
        outcome,
        group_col,
        label_col,
        holdout_fraction,
        seed,
        stratify=stratify,
        balance_by_pool_source=balance_by_pool_source,
        split_context=str(input_dir),
    )

    outcome_stays = validate_stays_partition(outcome, group_col, pretrain_stays, holdout_stays)
    if not holdout_stays:
        logger.warning("Holdout stay set is empty; downstream probe runs will have no data.")

    files = discover_parquet_files(input_dir, parquet_names)
    if not files:
        raise FileNotFoundError(f"No parquet files from {parquet_names} found in {input_dir}")

    for base, path in files:
        if base != outcome_basename:
            validate_segment_stays_subset(path, group_col, outcome_stays, base)

    output_pretrain_dir.mkdir(parents=True, exist_ok=True)
    output_holdout_dir.mkdir(parents=True, exist_ok=True)

    for base, path in files:
        filter_parquet_to_stays(path, group_col, pretrain_stays, output_pretrain_dir / base)
        filter_parquet_to_stays(path, group_col, holdout_stays, output_holdout_dir / base)

    manifest = build_manifest(
        seed,
        holdout_fraction,
        group_col,
        label_col,
        stratify,
        balance_by_pool_source,
        pretrain_stays,
        holdout_stays,
        outcome,
    )

    for out_root in (output_pretrain_dir, output_holdout_dir):
        with open(out_root / "holdout_split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

    return manifest
