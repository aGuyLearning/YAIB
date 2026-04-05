#!/usr/bin/env python3
"""Filter merged tri-corpus for SSL pretrain using hierarchical union holdout.

Scans single-site dirs ``data_root/<task>/<dataset>/outc.parquet``, computes holdout
per task×site, unions within each dataset (site), then unions to corpus level.
Equivalent to flat union over all pairs; manifest records per_dataset breakdown.

See docs/tri_corpus_holdout.md (hierarchical union section).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_YAIB_ROOT = Path(__file__).resolve().parents[2]
if str(_YAIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_YAIB_ROOT))

from icu_benchmarks.data.pooled_stay_id import pooled_index_map_from_merge_order
from icu_benchmarks.data.union_holdout import (
    discover_single_site_corpora,
    run_hierarchical_union_holdout_pretrain,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged-input-dir", type=Path, required=True, help="Merged corpus (e.g. corpus_eicu_hirid_miiv)")
    p.add_argument("--output-pretrain-dir", type=Path, required=True)
    p.add_argument(
        "--output-holdout-dir",
        type=Path,
        default=None,
        help="Optional: merged parquets restricted to union holdout stays (probe -d); manifest copy here too",
    )
    p.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Default: output-pretrain-dir/hierarchical_union_holdout_manifest.json",
    )
    p.add_argument("--data-root", type=Path, required=True, help="Repo data root (parent of task dirs)")
    p.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated task names (subdir under data-root)",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="eicu,hirid,miiv",
        help="Comma-separated site slugs (default: eicu,hirid,miiv)",
    )
    p.add_argument(
        "--require-all-pairs",
        action="store_true",
        help="Fail if any task×dataset pair lacks outc.parquet",
    )
    p.add_argument("--holdout-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42, help="Base seed; per-pair seed is derived")
    p.add_argument("--group-col", type=str, default="stay_id")
    p.add_argument("--label-col", type=str, default="label")
    p.add_argument("--no-stratify", action="store_true")
    p.add_argument("--balance-by-pool-source", action="store_true")
    p.add_argument(
        "--parquet-names",
        type=str,
        default="",
        help=(
            "Comma-separated basenames to copy. Default: all *.parquet files in the merged corpus "
            "directory, including valid/task masks and labels sidecars."
        ),
    )
    p.add_argument("--outcome-basename", type=str, default="outc.parquet")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any holdout stay id is missing from merged outcome",
    )
    p.add_argument(
        "--pooled-merge-order",
        type=str,
        default="eicu,hirid,miiv",
        help="Slugs matching merge_pooled_corpora --corpus-dirs order (default: tri build). "
        "Ignored with --no-pooled-stay-id-suffix.",
    )
    p.add_argument(
        "--no-pooled-stay-id-suffix",
        action="store_true",
        help="Use leaf outc stay_id values as-is (only if they already match merged corpus)",
    )
    args = p.parse_args()

    if not 0 < args.holdout_fraction < 1:
        p.error("holdout-fraction must be in (0, 1)")

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if not tasks:
        p.error("--tasks must list at least one task")
    if not datasets:
        p.error("--datasets must list at least one dataset")

    corpora = discover_single_site_corpora(
        args.data_root,
        tasks,
        datasets,
        require_all_pairs=args.require_all_pairs,
        outcome_basename=args.outcome_basename,
    )
    logger.info("Discovered %d task×dataset corpora under %s", len(corpora), args.data_root)

    names = tuple(x.strip() for x in args.parquet_names.split(",") if x.strip())
    manifest_path = args.manifest_path or (
        args.output_pretrain_dir / "hierarchical_union_holdout_manifest.json"
    )
    stratify = not args.no_stratify

    pooled_order: list[str] | None = None
    pooled_map: dict[str, int] | None = None
    if not args.no_pooled_stay_id_suffix:
        pooled_order = [x.strip() for x in args.pooled_merge_order.split(",") if x.strip()]
        if not pooled_order:
            p.error("--pooled-merge-order must list at least one slug when suffix remap is enabled")
        pooled_map = pooled_index_map_from_merge_order(pooled_order)

    manifest = run_hierarchical_union_holdout_pretrain(
        corpora,
        args.merged_input_dir,
        args.output_pretrain_dir,
        manifest_path,
        holdout_fraction=args.holdout_fraction,
        base_seed=args.seed,
        group_col=args.group_col,
        label_col=args.label_col,
        stratify=stratify,
        balance_by_pool_source=args.balance_by_pool_source,
        parquet_names=names,
        outcome_basename=args.outcome_basename,
        strict=args.strict,
        data_root=args.data_root,
        tasks_scanned=tasks,
        datasets_scanned=datasets,
        pooled_index_for_dataset=pooled_map,
        pooled_merge_order=pooled_order,
        output_holdout_dir=args.output_holdout_dir,
    )
    logger.info(
        "Wrote pretrain corpus excluding %d union holdout stays (merged had %d); manifest %s",
        manifest["n_union_holdout_stays"],
        manifest["n_merged_stays"],
        manifest_path,
    )
    if args.output_holdout_dir is not None:
        logger.info("Wrote holdout-only merged corpus under %s", args.output_holdout_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
