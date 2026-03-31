#!/usr/bin/env python3
"""Split a merged pooled YAIB corpus into pretrain vs global holdout directories.

See docs/tri_corpus_holdout.md for the full workflow (TS2Vec on P, TS2VecProbe + CV on H).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as script from repo root without install
_YAIB_ROOT = Path(__file__).resolve().parents[2]
if str(_YAIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_YAIB_ROOT))

from icu_benchmarks.data.corpus_split import DEFAULT_PARQUETS, run_corpus_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True, help="Merged corpus (outc/dyn/sta parquets)")
    p.add_argument("--output-pretrain-dir", type=Path, required=True)
    p.add_argument("--output-holdout-dir", type=Path, required=True)
    p.add_argument("--holdout-fraction", type=float, default=0.15, help="Fraction of stays for holdout")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-col", type=str, default="stay_id")
    p.add_argument("--label-col", type=str, default="label")
    p.add_argument(
        "--no-stratify",
        action="store_true",
        help="Random split even if label column exists",
    )
    p.add_argument(
        "--balance-by-pool-source",
        action="store_true",
        help="Split each pool-source bucket (PooledData stay_id suffix) separately with the same holdout fraction",
    )
    p.add_argument(
        "--parquet-names",
        type=str,
        default=",".join(DEFAULT_PARQUETS),
        help="Comma-separated basenames to copy (only existing files are used)",
    )
    p.add_argument("--outcome-basename", type=str, default="outc.parquet")
    args = p.parse_args()

    if not 0 < args.holdout_fraction < 1:
        p.error("holdout-fraction must be in (0, 1)")

    names = tuple(x.strip() for x in args.parquet_names.split(",") if x.strip())
    stratify = not args.no_stratify

    manifest = run_corpus_split(
        args.input_dir,
        args.output_pretrain_dir,
        args.output_holdout_dir,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        group_col=args.group_col,
        label_col=args.label_col,
        stratify=stratify,
        balance_by_pool_source=args.balance_by_pool_source,
        parquet_names=names,
        outcome_basename=args.outcome_basename,
    )
    logger.info(
        "Wrote pretrain (%d stays) and holdout (%d stays); manifest in each output dir.",
        manifest["n_stays_pretrain"],
        manifest["n_stays_holdout"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
