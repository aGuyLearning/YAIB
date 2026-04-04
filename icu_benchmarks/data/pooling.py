"""Offline pooling of per-site YAIB parquet corpora with optional subsampling.

Mirrors the stay_id suffix scheme used by ``merge_pooled_corpora.py``:
``pooled_id = int(str(original_id) + repeated_digit)`` with ``repeated_digit = "1111"``, ``"2222"``, …

Typical layout: ``data_dir`` contains subfolders named ``eicu``, ``hirid``, ``miiv`` each with
``dyn.parquet``, ``sta.parquet``, ``outc.parquet``. ``file_names`` maps
:class:`DataSegment` keys to those basenames.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from icu_benchmarks.constants import RunMode

from .constants import DataSegment as Segment
from .constants import VarType as Var


class PooledDataset:
    """Preset ``datasets`` lists for :class:`PooledData` (eicu, hirid, miiv only).

    Pooled output folder names use ``sorted(site_names)`` (e.g. triple → ``eicu_hirid_miiv``).
    """

    SITES = ("eicu", "hirid", "miiv")
    eicu_hirid = ["eicu", "hirid"]
    eicu_miiv = ["eicu", "miiv"]
    hirid_miiv = ["hirid", "miiv"]
    hirid_eicu_miiv = ["hirid", "eicu", "miiv"]


class PooledData:
    def __init__(
        self,
        data_dir: Path | str,
        vars: dict,
        datasets: list[str],
        file_names: dict,
        shuffle: bool = False,
        stratify=None,
        runmode: RunMode = RunMode.classification,
        save_test: bool = True,
    ):
        """
        Args:
            data_dir: Parent directory with one subfolder per site (e.g. eicu, hirid).
            vars: YAIB-style vars dict with GROUP, LABEL (and optionally SEQUENCE).
            datasets: Folder names under data_dir to include (must exist).
            file_names: Maps Segment.static/dynamic/outcome to parquet basenames.
            shuffle: Passed to train_test_split.
            stratify: Unused legacy; stratification uses labels derived from outcome.
            runmode: Classification enables stratified split by label when possible.
            save_test: If True, write held-out stays per site before pooling train subset.
        """
        self.data_dir = Path(data_dir)
        self.vars = vars
        self.datasets = list(datasets)
        self.file_names = file_names
        self.shuffle = shuffle
        self.stratify = stratify
        self.runmode = runmode
        self.save_test = save_test

    def generate(
        self,
        datasets: list[str] | None = None,
        samples: int = 10_000,
        seed: int = 42,
    ) -> None:
        """Load site folders, subsample stays, suffix stay_ids, concat, write pooled parquets."""
        use_datasets = list(datasets) if datasets is not None else self.datasets
        loaded: dict[str, dict[str, pd.DataFrame]] = {}
        for name in sorted(use_datasets):
            folder = self.data_dir / name
            if not folder.is_dir():
                continue
            loaded[name] = {
                seg: pq.read_table(folder / self.file_names[seg]).to_pandas(self_destruct=True)
                for seg in self.file_names
            }
        missing = [n for n in sorted(use_datasets) if n not in loaded]
        if missing:
            raise FileNotFoundError(
                f"Missing or not a directory under {self.data_dir}: {missing}. "
                f"Expected subfolders with {list(self.file_names.values())}."
            )

        pooled = self._pool_datasets(
            datasets=loaded,
            samples=samples,
            vars_dict=self.vars,
            shuffle=self.shuffle,
            seed=seed,
            runmode=self.runmode,
            data_dir=self.data_dir,
            save_test=self.save_test,
        )
        self._save_pooled_data(self.data_dir, pooled, use_datasets, self.file_names, samples=samples)

    def _save_pooled_data(
        self,
        data_dir: Path,
        data: dict[str, pd.DataFrame],
        datasets: list[str],
        file_names: dict,
        samples: int = 10_000,
    ) -> None:
        save_folder = "_".join(sorted(datasets))
        save_folder += f"_{samples}"
        save_dir = data_dir / save_folder
        save_dir.mkdir(parents=True, exist_ok=True)
        for key, frame in data.items():
            frame.to_parquet(save_dir / Path(file_names[key]), index=False)
        logging.info("Saved pooled data at %s", save_dir)

    def _pool_datasets(
        self,
        datasets: dict[str, dict[str, pd.DataFrame]] | None,
        samples: int,
        vars_dict: dict,
        shuffle: bool,
        seed: int,
        runmode: RunMode,
        data_dir: Path,
        save_test: bool,
    ) -> dict[str, pd.DataFrame]:
        if not datasets:
            raise ValueError("No datasets supplied.")
        group_col = vars_dict[Var.group]
        label_col = vars_dict[Var.label]

        pooled_data: dict[str, list[pd.DataFrame]] = {
            Segment.static: [],
            Segment.dynamic: [],
            Segment.outcome: [],
        }
        site_index = 0
        for key, value in datasets.items():
            site_index += 1
            repeated_digit = str(site_index) * 4
            outcome = value[Segment.outcome].copy()
            static = value[Segment.static].copy()
            dynamic = value[Segment.dynamic].copy()

            stays = pd.Series(outcome[group_col].unique(), dtype="int64")
            n_stays = len(stays)
            if n_stays == 0:
                raise ValueError(f"No stays in outcome for dataset {key}")

            n_train = min(int(samples), n_stays)
            if n_train < 1:
                raise ValueError("samples must be >= 1")

            stratify_y = None
            if runmode is RunMode.classification:
                label_by_stay = outcome.groupby(group_col, sort=False)[label_col].max()
                aligned = label_by_stay.reindex(stays.values).to_numpy()
                if len(pd.unique(aligned)) > 1:
                    counts = pd.Series(aligned).value_counts()
                    if int(counts.min()) >= 2:
                        stratify_y = aligned

            # sklearn requires train_size < n_samples; use all stays as train when capped.
            if n_train >= n_stays:
                stays_train = stays.values.copy()
                stays_test = stays.values[:0]
            else:
                # Stratified split requires shuffle=True in sklearn.
                effective_shuffle = shuffle if stratify_y is None else True
                stays_train, stays_test = train_test_split(
                    stays.values,
                    train_size=n_train,
                    shuffle=effective_shuffle,
                    random_state=seed,
                    stratify=stratify_y,
                )

            if save_test and len(stays_test) > 0:
                o_te, s_te, d_te = _select_stays(
                    outcome, static, dynamic, stays_test, repeated_digit, group_col
                )
                test_dir = data_dir / f"{key}_test_{len(stays_test)}"
                test_dir.mkdir(parents=True, exist_ok=True)
                o_te.to_parquet(test_dir / "outc.parquet", index=False)
                s_te.to_parquet(test_dir / "sta.parquet", index=False)
                d_te.to_parquet(test_dir / "dyn.parquet", index=False)
                logging.info("Saved held-out site data at %s", test_dir)

            o_tr, s_tr, d_tr = _select_stays(
                outcome, static, dynamic, stays_train, repeated_digit, group_col
            )
            pooled_data[Segment.static].append(s_tr)
            pooled_data[Segment.dynamic].append(d_tr)
            pooled_data[Segment.outcome].append(o_tr)

        return {k: pd.concat(v, ignore_index=True) for k, v in pooled_data.items()}


def _select_stays(
    outcome: pd.DataFrame,
    static: pd.DataFrame,
    dynamic: pd.DataFrame,
    select,
    repeated_digit: str,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sel = set(int(x) for x in select)
    o = outcome.loc[outcome[group_col].isin(sel)].copy()
    s = static.loc[static[group_col].isin(sel)].copy()
    d = dynamic.loc[dynamic[group_col].isin(sel)].copy()
    for frame in (o, s, d):
        frame[group_col] = frame[group_col].map(lambda x: int(str(int(x)) + repeated_digit))
    return o, s, d
