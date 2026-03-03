#!/usr/bin/env python3
"""Extract and visualize TS2Vec representations from a pretrained checkpoint."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import gin
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from sklearn.decomposition import PCA

from icu_benchmarks.constants import RunMode
from icu_benchmarks.cross_validation import execute_repeated_cv  # noqa: F401
from icu_benchmarks.data.constants import DataSegment, DataSplit
from icu_benchmarks.data.loader import PretrainPolarsDataset
from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.models.dl_models.ts2vec import TS2Vec
from icu_benchmarks.run import get_mode  # noqa: F401

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TS2Vec embeddings from YAIB demo corpus and visualize with PCA."
    )
    parser.add_argument("--yaib-root", type=Path, default=Path("."), help="Path to YAIB repository root.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to pretrain corpus (e.g., demo_data/corpus_mimic_demo).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to TS2Vec checkpoint (last.ckpt/model.ckpt).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../yaib_logs_ts2vec_kickoff/representations"),
        help="Directory where outputs are written.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--max-samples", type=int, default=600, help="Max number of stays to plot.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_configs(yaib_root: Path) -> None:
    gin.clear_config()
    gin.parse_config_file(str(yaib_root / "configs/tasks/Pretrain.gin"))
    gin.parse_config_file(str(yaib_root / "configs/prediction_models/TS2Vec.gin"))


def extract_representations(data: dict, ckpt_path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    dataset = PretrainPolarsDataset(data, split=DataSplit.train, ram_cache=True)
    stay_ids = dataset.grouping_df["stay_id"].unique().to_list()
    windows = torch.stack([dataset[i] for i in range(len(dataset))], dim=0).float()
    windows = torch.nan_to_num(windows, nan=0.0, posinf=0.0, neginf=0.0)

    model = TS2Vec.load_from_checkpoint(str(ckpt_path), map_location="cpu", weights_only=False)
    model.eval()
    with torch.no_grad():
        reps = model.encode(windows).mean(dim=1).cpu().numpy()

    labels_df = data[DataSplit.train][DataSegment.outcome]
    labels_map = (
        labels_df.sort("time")
        .group_by("stay_id")
        .agg(pl.col("label").drop_nulls().last().alias("label"))
        .to_dict(as_series=False)
    )
    labels_lookup = dict(zip(labels_map["stay_id"], labels_map["label"]))
    labels = np.array([labels_lookup.get(stay_id, np.nan) for stay_id in stay_ids], dtype=np.float32)
    return reps, stay_ids, labels


def maybe_subsample(
    reps: np.ndarray, stay_ids: list[str], labels: np.ndarray, max_samples: int, seed: int
) -> tuple[np.ndarray, list[str], np.ndarray]:
    if reps.shape[0] <= max_samples:
        return reps, stay_ids, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(reps.shape[0], size=max_samples, replace=False)
    idx.sort()
    return reps[idx], [stay_ids[i] for i in idx], labels[idx]


def save_outputs(out_dir: Path, reps: np.ndarray, pcs: np.ndarray, stay_ids: list[str], labels: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ts2vec_representations.npy", reps)

    table = pl.DataFrame(
        {
            "stay_id": stay_ids,
            "label": labels,
            "pc1": pcs[:, 0],
            "pc2": pcs[:, 1],
        }
    )
    table.write_csv(out_dir / "ts2vec_representations_pca.csv")

    valid_label_mask = np.isfinite(labels)
    if valid_label_mask.any():
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(pcs[:, 0], pcs[:, 1], c=np.where(valid_label_mask, labels, -1), s=18, alpha=0.75, cmap="viridis")
        plt.colorbar(scatter, label="Label")
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(pcs[:, 0], pcs[:, 1], s=18, alpha=0.75, color="steelblue")
    plt.title("TS2Vec Representation PCA (demo corpus)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_dir / "ts2vec_representation_pca.png", dpi=180)
    plt.close()


def main() -> int:
    args = parse_args()
    yaib_root = args.yaib_root.resolve()
    data_dir = args.data_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    out_dir = args.out_dir.resolve()
    mplconfigdir = yaib_root / ".mplconfig"
    mplconfigdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfigdir))

    set_seed(args.seed)
    parse_configs(yaib_root)

    data = preprocess_data(
        data_dir=data_dir,
        seed=args.seed,
        debug=False,
        load_cache=False,
        generate_cache=False,
        cv_repetitions=5,
        repetition_index=0,
        cv_folds=5,
        fold_index=0,
        train_size=None,
        complete_train=True,
        runmode=RunMode.pretrain,
    )

    reps, stay_ids, labels = extract_representations(data, checkpoint)
    reps, stay_ids, labels = maybe_subsample(reps, stay_ids, labels, args.max_samples, args.seed)
    pcs = PCA(n_components=2, random_state=args.seed).fit_transform(reps)
    save_outputs(out_dir, reps, pcs, stay_ids, labels)
    print(f"Saved representations and PCA plot to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
