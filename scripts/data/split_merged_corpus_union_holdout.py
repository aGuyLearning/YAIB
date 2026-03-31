#!/usr/bin/env python3
"""Filter a merged corpus for SSL pretrain using the union of per-(task,dataset) holdouts.

Each --pair (or JSON entry) points to a directory with its own outc.parquet. Holdout
stay IDs are computed the same way as split_tri_corpus_holdout.py, then unioned.
The merged corpus is written with only stays outside that union.

See docs/tri_corpus_holdout.md (union holdout section).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_YAIB_ROOT = Path(__file__).resolve().parents[2]
if str(_YAIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_YAIB_ROOT))

from icu_benchmarks.data.corpus_split import DEFAULT_PARQUETS
from icu_benchmarks.data.pooled_stay_id import pooled_index_map_from_merge_order
from icu_benchmarks.data.union_holdout import (
    corpora_from_json,
    parse_pair_arg,
    run_union_holdout_pretrain,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged-input-dir", type=Path, required=True, help="Merged corpus (outc/dyn/sta parquets)")
    p.add_argument("--output-pretrain-dir", type=Path, required=True)
    p.add_argument(
        "--output-holdout-dir",
        type=Path,
        default=None,
        help="Optional: write merged rows for union holdout stays only (for probe -d); "
        "also writes a copy of the manifest here",
    )
    p.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Defaults to output-pretrain-dir/union_holdout_manifest.json",
    )
    p.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="TASK=DATASET=PATH",
        help="Repeatable; PATH is directory containing outcome parquet",
    )
    p.add_argument(
        "--pairs-json",
        type=Path,
        default=None,
        help='JSON list of {"task","dataset","path"} objects (alternative to --pair)',
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
        default=",".join(DEFAULT_PARQUETS),
        help="Comma-separated basenames to copy (only existing files are used)",
    )
    p.add_argument("--outcome-basename", type=str, default="outc.parquet")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any pair holdout stay id is missing from merged outcome",
    )
    p.add_argument(
        "--pooled-merge-order",
        type=str,
        default=None,
        help="If set (comma slugs), remap leaf holdout stay_ids to merged pooled space; "
        "each pair's dataset field must appear in this order list.",
    )
    p.add_argument(
        "--no-pooled-stay-id-suffix",
        action="store_true",
        help="Disable suffix remap even if --pooled-merge-order is set",
    )
    args = p.parse_args()

    if not 0 < args.holdout_fraction < 1:
        p.error("holdout-fraction must be in (0, 1)")

    if bool(args.pairs_json) == bool(args.pair):
        p.error("Provide exactly one of --pairs-json or one or more --pair")

    if args.pairs_json:
        corpora = corpora_from_json(args.pairs_json)
    else:
        corpora = [parse_pair_arg(s) for s in args.pair]

    names = tuple(x.strip() for x in args.parquet_names.split(",") if x.strip())
    manifest_path = args.manifest_path or (args.output_pretrain_dir / "union_holdout_manifest.json")
    stratify = not args.no_stratify

    pooled_order: list[str] | None = None
    pooled_map: dict[str, int] | None = None
    if not args.no_pooled_stay_id_suffix and args.pooled_merge_order:
        pooled_order = [x.strip() for x in args.pooled_merge_order.split(",") if x.strip()]
        if not pooled_order:
            p.error("--pooled-merge-order must list at least one slug when set")
        pooled_map = pooled_index_map_from_merge_order(pooled_order)
        for c in corpora:
            if c.dataset not in pooled_map:
                p.error(
                    f"Pair {c.pair_key()}: dataset slug {c.dataset!r} not in --pooled-merge-order"
                )

    manifest = run_union_holdout_pretrain(
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
        pooled_index_for_dataset=pooled_map,
        pooled_merge_order=pooled_order,
        output_holdout_dir=args.output_holdout_dir,
    )
    logger.info(
        "Wrote pretrain corpus excluding %d union holdout stays (merged had %d stays); manifest %s",
        manifest["n_union_holdout_stays"],
        manifest["n_merged_stays"],
        manifest_path,
    )
    if args.output_holdout_dir is not None:
        logger.info("Wrote holdout-only merged corpus under %s", args.output_holdout_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
