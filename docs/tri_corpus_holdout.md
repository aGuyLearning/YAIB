# Tri-corpus global holdout (pretrain vs probe)

Use this when you merge multiple ICU sources into one pooled corpus (see `icu_benchmarks.data.pooling.PooledData`) and want:

1. **Pretraining** (TS2Vec) only on a **pretrain pool** `P` of stays.
2. A **global holdout** `H` disjoint from `P` for **downstream supervised probes** with YAIB cross-validation.
3. Optional comparison of **standard vs homogeneous-dataset batching** during pretrain (same `P`, same `H`, two checkpoints).

YAIB’s `preprocess_data` / `make_single_split_polars` only split stays **inside** the directory you pass to `-d`. It does not define a corpus-level holdout across “all merged data then pretrain on a subset”. This workflow **materializes two directories** so leakage is explicit.

## 1. Build the merged corpus

Produce a single folder containing at least `outc.parquet`, `dyn.parquet`, and optionally `sta.parquet` (same layout as task gin `preprocess.file_names`).

## 2. Split into pretrain and holdout

Use the **`yaib`** conda environment (or any env with `polars`, `scikit-learn`, and YAIB deps such as `torch`). From the YAIB repo root:

```bash
conda activate yaib
# or: conda run -n yaib --no-capture-output python ...
python scripts/data/split_tri_corpus_holdout.py \
  --input-dir /path/to/merged_eicu_hirid_miiv_10000 \
  --output-pretrain-dir /path/to/tri_pretrain \
  --output-holdout-dir /path/to/tri_holdout \
  --holdout-fraction 0.15 \
  --seed 42
```

Options:

- **`--no-stratify`**: random stay split even if `label` exists.
- **`--balance-by-pool-source`**: for each pool-source bucket (suffix on `stay_id` from `PooledData`, same convention as `pool_source_id_from_stay_id`), apply the same holdout fraction, then concatenate. Helps avoid emptying a small source in the holdout.

Each output directory gets the same parquet basenames (filtered rows) plus `holdout_split_manifest.json` (stay lists, label and pool-source counts).

**Gin alignment:** Pretrain often uses `preprocess.use_static = False`; if your merged folder has no `sta.parquet`, match that in gin. Holdout/probe tasks that need static features must include `sta` in the merged input before splitting.

### 2.1 Union holdout across many (task, dataset) corpora

When one SSL pretrain run must support **several** downstream probes (different tasks and/or per-site pools), a single random split on the **merged** `outc` is not aligned with each task’s cohort. Instead:

1. For **each** (task, dataset) layout you will probe, build holdout stay IDs from **that** directory’s `outc.parquet` using the same rules as above (`holdout_fraction`, stratify, optional `balance_by_pool_source`). Each pair only sees stays that exist for **that** task’s labels.
2. Take the **set union** of all those holdout stay IDs.
3. Filter the **merged** tri-corpus to **exclude** that union for pretrain only.

Implementation: `icu_benchmarks.data.union_holdout`, CLI `scripts/data/split_merged_corpus_union_holdout.py`.

```bash
conda activate yaib
python scripts/data/split_merged_corpus_union_holdout.py \
  --merged-input-dir /path/to/merged_tri \
  --output-pretrain-dir /path/to/tri_pretrain_union \
  --holdout-fraction 0.15 \
  --seed 42 \
  --pair mortality24=eicu=/path/to/data/mortality24/eicu_pool \
  --pair aki=miiv=/path/to/data/aki/miiv_pool
```

Or pass `--pairs-json /path/to/pairs.json` with a list of `{"task", "dataset", "path"}` objects (each `path` is a directory containing `outc.parquet`).

**`union_holdout_manifest.json`** (written next to the pretrain output by default) includes:

- `union_holdout_stay_ids`: sorted union used for exclusion.
- `per_pair`: each `task::dataset` key maps to `holdout_stay_ids` and `n_holdout_stays`.
- Counts: `n_merged_stays`, `n_union_holdout_stays`, `n_extra_union_not_in_merged` (non-zero if a pair’s holdout id does not appear in merged outcome; use `--strict` to fail instead of warning).

**Probes** still use task-specific `-d` data with only **relevant stays**; the manifest records which stays were globally withheld from SSL.

**Tests:** from the Bachelor repo root (or any parent of `YAIB/` on `PYTHONPATH`):

```bash
conda activate yaib
cd /path/to/Bachelor
PYTHONPATH=YAIB python -m pytest YAIB/tests/test_union_holdout.py -v
```

## 3. Pretrain TS2Vec twice (batching ablation)

Point `-d` at **`tri_pretrain` only**. Keep gin identical except:

- Run A: `train_common.homogeneous_dataset_batches = False` (default).
- Run B: `train_common.homogeneous_dataset_batches = True`.

Use the same seed, epochs, and model gin (e.g. `configs/prediction_models/TS2Vec.gin` + Pretrain task). Save two checkpoints (e.g. `last.ckpt` under two run names).

## 4. Train probes on the holdout with CV

Point `-d` at **`tri_holdout` only**. Use a classification (or regression) task gin with **`TS2VecProbe`** and set `TS2VecProbe.pretrained_encoder_path` to each pretrain checkpoint in turn. Use normal repeated CV (`configs/tasks/common/CrossValidation.gin`).

See `scripts/sample_usage/transfer/ts2vec_kickoff.py` for patterns that inject `TS2VecProbe.pretrained_encoder_path` and match preprocessor settings to pretrain.

## 5. Validation rules

The script requires:

- Every `stay_id` in `outc.parquet` is assigned to exactly one of `P` or `H`.
- Other segments (`dyn`, `sta`) must not introduce `stay_id`s that are absent from `outc.parquet` (otherwise rows would be lost when filtering).

## Reference

- Pooling: `icu_benchmarks/data/pooling.py`
- Homogeneous batching: `icu_benchmarks/data/batch_samplers.py`, `train_common.homogeneous_dataset_batches` (Pretrain + TS2Vec only)
- Split implementation: `icu_benchmarks/data/corpus_split.py`
- Union holdout: `icu_benchmarks/data/union_holdout.py`
- CLIs: `scripts/data/split_tri_corpus_holdout.py`, `scripts/data/split_merged_corpus_union_holdout.py`
