"""Tests for union holdout pretrain filtering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

_YAIB_ROOT = Path(__file__).resolve().parents[1]
if str(_YAIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_YAIB_ROOT))

from icu_benchmarks.data.pooled_stay_id import (
    apply_pooled_stay_id_suffix,
    pooled_index_map_from_merge_order,
    pooled_suffix_digits,
)
from icu_benchmarks.data.union_holdout import (
    TaskDatasetCorpus,
    compute_hierarchical_union_holdout,
    compute_union_holdout,
    discover_single_site_corpora,
    merged_outcome_stays,
    run_union_holdout_pretrain,
    validate_union_against_merged,
    write_merged_holdout_subset,
    write_pretrain_excluding_stays,
)


def _write_outc(path: Path, stay_ids: list[int], labels: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if labels is None:
        labels = [i % 2 for i in range(len(stay_ids))]
    pl.DataFrame({"stay_id": stay_ids, "label": labels[: len(stay_ids)]}).write_parquet(path)


def _write_dyn(path: Path, stay_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in stay_ids:
        rows.append({"stay_id": sid, "time": 0.0, "x": float(sid)})
    pl.DataFrame(rows).write_parquet(path)


def test_union_equals_set_union_of_pair_holdouts(tmp_path: Path):
    """Union is exactly the set union of per-pair holdouts (dedupes shared stays)."""
    p1 = tmp_path / "m_eicu"
    p2 = tmp_path / "m_mimic"
    shared = 103
    _write_outc(p1 / "outc.parquet", [101, 102, shared, 104], [0, 1, 0, 1])
    _write_outc(p2 / "outc.parquet", [201, 202, shared], [0, 1, 1])

    corpora = [
        TaskDatasetCorpus("mortality24", "eicu", p1),
        TaskDatasetCorpus("mortality24", "mimic", p2),
    ]
    union_h, per_pair = compute_union_holdout(
        corpora,
        holdout_fraction=0.25,
        base_seed=42,
        group_col="stay_id",
        label_col="label",
        stratify=False,
        balance_by_pool_source=False,
    )
    h1 = per_pair["mortality24::eicu"]
    h2 = per_pair["mortality24::mimic"]
    assert len(union_h) <= len(h1) + len(h2)
    assert union_h == h1 | h2


def test_pretrain_excludes_union(tmp_path: Path):
    merged = tmp_path / "merged"
    stays = list(range(1, 21))
    _write_outc(merged / "outc.parquet", stays)
    _write_dyn(merged / "dyn.parquet", stays)

    pair_dir = tmp_path / "pair_a"
    _write_outc(pair_dir / "outc.parquet", stays[:15])
    corpora = [TaskDatasetCorpus("t1", "d1", pair_dir)]

    union_h, _ = compute_union_holdout(
        corpora,
        holdout_fraction=0.2,
        base_seed=123,
        group_col="stay_id",
        label_col="label",
        stratify=False,
        balance_by_pool_source=False,
    )
    assert union_h

    out_pt = tmp_path / "pretrain"
    write_pretrain_excluding_stays(
        merged,
        out_pt,
        union_h,
        group_col="stay_id",
        parquet_names=("outc.parquet", "dyn.parquet"),
    )

    pt_outc = pl.read_parquet(out_pt / "outc.parquet")
    pt_stays = set(pt_outc["stay_id"].to_list())
    assert not (pt_stays & union_h)
    assert pt_stays == set(stays) - union_h


def test_determinism_same_seed(tmp_path: Path):
    d = tmp_path / "p"
    _write_outc(d / "outc.parquet", list(range(50, 90)), [i % 2 for i in range(40)])
    corpora = [TaskDatasetCorpus("a", "b", d)]
    u1, _ = compute_union_holdout(
        corpora,
        holdout_fraction=0.15,
        base_seed=7,
        group_col="stay_id",
        label_col="label",
        stratify=True,
        balance_by_pool_source=False,
    )
    u2, _ = compute_union_holdout(
        corpora,
        holdout_fraction=0.15,
        base_seed=7,
        group_col="stay_id",
        label_col="label",
        stratify=True,
        balance_by_pool_source=False,
    )
    assert u1 == u2


def test_strict_raises_when_union_not_subset_of_merged():
    """Deterministic: holdout set is not random—must contain an id missing from merged."""
    mstays = {1, 2, 3}
    union_h = {1, 9999}
    with pytest.raises(ValueError, match="holdout stay id"):
        validate_union_against_merged(union_h, mstays, strict=True)

    validate_union_against_merged(union_h, mstays, strict=False)


def test_run_union_holdout_pretrain_strict_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pair = tmp_path / "pair"
    _write_outc(pair / "outc.parquet", [10, 11, 12], [0, 1, 0])
    merged = tmp_path / "merged"
    _write_outc(merged / "outc.parquet", [10, 11, 12])
    _write_dyn(merged / "dyn.parquet", [10, 11, 12])

    import icu_benchmarks.data.union_holdout as uh

    def fake_compute(
        corpora: object,
        **kwargs: object,
    ) -> tuple[set[int], dict[str, set[int]]]:
        return {88888}, {"x::y": {88888}}

    monkeypatch.setattr(uh, "compute_union_holdout", fake_compute)

    manifest_p = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="holdout stay id"):
        run_union_holdout_pretrain(
            [TaskDatasetCorpus("x", "y", pair)],
            merged,
            tmp_path / "pt",
            manifest_p,
            holdout_fraction=0.33,
            base_seed=1,
            group_col="stay_id",
            label_col="label",
            stratify=False,
            balance_by_pool_source=False,
            strict=True,
        )


def test_single_stay_pair_yields_empty_holdout_union(tmp_path: Path):
    pair = tmp_path / "one"
    _write_outc(pair / "outc.parquet", [42], [0])
    union_h, per = compute_union_holdout(
        [TaskDatasetCorpus("t", "d", pair)],
        holdout_fraction=0.15,
        base_seed=42,
        group_col="stay_id",
        label_col="label",
        stratify=True,
        balance_by_pool_source=False,
    )
    assert union_h == set()
    assert per["t::d"] == set()

    merged = tmp_path / "merged"
    _write_outc(merged / "outc.parquet", [40, 41, 42])
    _write_dyn(merged / "dyn.parquet", [40, 41, 42])
    out_pt = tmp_path / "pretrain"
    write_pretrain_excluding_stays(
        merged,
        out_pt,
        union_h,
        group_col="stay_id",
        parquet_names=("outc.parquet", "dyn.parquet"),
    )
    assert pl.read_parquet(out_pt / "outc.parquet").height == 3


def test_run_union_holdout_writes_holdout_corpus_and_manifest(tmp_path: Path):
    pair = tmp_path / "pair"
    merged_stays_list = list(range(1, 21))
    _write_outc(pair / "outc.parquet", merged_stays_list[:12], [i % 2 for i in range(12)])
    merged = tmp_path / "merged"
    _write_outc(merged / "outc.parquet", merged_stays_list)
    _write_dyn(merged / "dyn.parquet", merged_stays_list)
    pre = tmp_path / "pretrain"
    hold = tmp_path / "holdout"
    mpath = tmp_path / "union_manifest.json"
    run_union_holdout_pretrain(
        [TaskDatasetCorpus("aki", "eicu", pair)],
        merged,
        pre,
        mpath,
        holdout_fraction=0.2,
        base_seed=99,
        group_col="stay_id",
        label_col="label",
        stratify=False,
        balance_by_pool_source=False,
        strict=True,
        output_holdout_dir=hold,
    )
    pt_ids = set(pl.read_parquet(pre / "outc.parquet")["stay_id"].to_list())
    ho_ids = set(pl.read_parquet(hold / "outc.parquet")["stay_id"].to_list())
    assert not (pt_ids & ho_ids)
    assert pt_ids | ho_ids == set(merged_stays_list)
    assert (hold / mpath.name).is_file()


def test_write_merged_holdout_subset_intersection(tmp_path: Path):
    merged = tmp_path / "merged"
    _write_outc(merged / "outc.parquet", [1, 2, 3, 4])
    _write_dyn(merged / "dyn.parquet", [1, 2, 3, 4])
    out_h = tmp_path / "ho"
    write_merged_holdout_subset(merged, out_h, {2, 3, 99}, group_col="stay_id", parquet_names=("outc.parquet", "dyn.parquet"))
    assert set(pl.read_parquet(out_h / "outc.parquet")["stay_id"].to_list()) == {2, 3}


def test_run_union_holdout_writes_manifest(tmp_path: Path):
    pair = tmp_path / "pair"
    _write_outc(pair / "outc.parquet", list(range(1, 25)), [i % 2 for i in range(24)])
    merged = tmp_path / "merged"
    all_s = list(range(1, 31))
    _write_outc(merged / "outc.parquet", all_s)
    _write_dyn(merged / "dyn.parquet", all_s)

    mpath = tmp_path / "u.json"
    run_union_holdout_pretrain(
        [TaskDatasetCorpus("aki", "eicu", pair)],
        merged,
        tmp_path / "pretrain_out",
        mpath,
        holdout_fraction=0.2,
        base_seed=99,
        group_col="stay_id",
        label_col="label",
        stratify=False,
        balance_by_pool_source=False,
        strict=True,
    )
    data = json.loads(mpath.read_text(encoding="utf-8"))
    assert data["kind"] == "union_holdout_pretrain"
    assert "per_pair" in data
    assert data["n_extra_union_not_in_merged"] == 0


def test_discover_single_site_corpora_skips_missing(tmp_path: Path):
    root = tmp_path / "data"
    _write_outc(root / "t1" / "eicu" / "outc.parquet", list(range(10, 30)), [i % 2 for i in range(20)])
    found = discover_single_site_corpora(
        root,
        ["t1", "t2"],
        ["eicu", "miiv"],
        require_all_pairs=False,
    )
    assert len(found) == 1
    assert found[0].pair_key() == "t1::eicu"


def test_discover_single_site_corpora_require_all_raises(tmp_path: Path):
    root = tmp_path / "data"
    _write_outc(root / "t1" / "eicu" / "outc.parquet", [1, 2], [0, 1])
    with pytest.raises(FileNotFoundError, match="require_all_pairs"):
        discover_single_site_corpora(
            root,
            ["t1", "t2"],
            ["eicu"],
            require_all_pairs=True,
        )


def test_pooled_suffix_helpers_match_merge_pooled_corpora():
    assert pooled_suffix_digits(0) == "1111"
    assert pooled_suffix_digits(2) == "3333"
    assert apply_pooled_stay_id_suffix(100, 0) == int("100" + "1111")
    assert apply_pooled_stay_id_suffix(5, 1) == int("5" + "2222")
    m = pooled_index_map_from_merge_order(["eicu", "hirid", "miiv"])
    assert m == {"eicu": 0, "hirid": 1, "miiv": 2}


def test_pooled_index_map_duplicate_slug_raises():
    with pytest.raises(ValueError, match="Duplicate"):
        pooled_index_map_from_merge_order(["eicu", "eicu"])


def test_pooled_remap_matches_manual_apply_per_stay(tmp_path: Path):
    pair = tmp_path / "p"
    stays = list(range(100, 120))
    _write_outc(pair / "outc.parquet", stays, [i % 2 for i in range(20)])
    corpora = [TaskDatasetCorpus("t", "eicu", pair)]
    kwargs = dict(
        holdout_fraction=0.25,
        base_seed=3,
        group_col="stay_id",
        label_col="label",
        stratify=False,
        balance_by_pool_source=False,
    )
    raw_u, raw_p = compute_union_holdout(corpora, **kwargs, pooled_index_for_dataset=None)
    pmap = pooled_index_map_from_merge_order(["eicu"])
    u2, p2 = compute_union_holdout(corpora, **kwargs, pooled_index_for_dataset=pmap)
    assert u2 == {apply_pooled_stay_id_suffix(s, 0) for s in raw_u}
    assert p2["t::eicu"] == {apply_pooled_stay_id_suffix(s, 0) for s in raw_p["t::eicu"]}


def test_dataset_missing_from_pooled_map_raises(tmp_path: Path):
    pair = tmp_path / "p"
    _write_outc(pair / "outc.parquet", [1, 2, 3, 4], [0, 1, 0, 1])
    with pytest.raises(ValueError, match="not in pooled merge-order map"):
        compute_union_holdout(
            [TaskDatasetCorpus("t", "miiv", pair)],
            holdout_fraction=0.25,
            base_seed=0,
            group_col="stay_id",
            label_col="label",
            stratify=False,
            balance_by_pool_source=False,
            pooled_index_for_dataset={"eicu": 0},
        )


def test_write_pretrain_excludes_suffixed_stay_id(tmp_path: Path):
    merged = tmp_path / "merged"
    sid = apply_pooled_stay_id_suffix(42, 0)
    _write_outc(merged / "outc.parquet", [sid])
    _write_dyn(merged / "dyn.parquet", [sid])
    out_pt = tmp_path / "out"
    write_pretrain_excluding_stays(
        merged,
        out_pt,
        {sid},
        group_col="stay_id",
        parquet_names=("outc.parquet", "dyn.parquet"),
    )
    assert pl.read_parquet(out_pt / "outc.parquet").height == 0


def test_hierarchical_corpus_union_equals_flat_union(tmp_path: Path):
    root = tmp_path / "data"
    for task, offset in [("t1", 0), ("t2", 1000)]:
        for ds, add in [("eicu", 0), ("miiv", 500)]:
            stays = list(range(offset + add, offset + add + 40))
            _write_outc(root / task / ds / "outc.parquet", stays, [i % 2 for i in range(40)])

    corpora = discover_single_site_corpora(root, ["t1", "t2"], ["eicu", "miiv"])
    flat_u, per_flat = compute_union_holdout(
        corpora,
        holdout_fraction=0.2,
        base_seed=11,
        group_col="stay_id",
        label_col="label",
        stratify=False,
        balance_by_pool_source=False,
    )
    hier = compute_hierarchical_union_holdout(
        corpora,
        holdout_fraction=0.2,
        base_seed=11,
        group_col="stay_id",
        label_col="label",
        stratify=False,
        balance_by_pool_source=False,
    )
    assert hier.corpus_union == flat_u
    assert hier.per_pair == per_flat
    # per-dataset = union of task holdouts for that site
    ue = hier.per_pair["t1::eicu"] | hier.per_pair["t2::eicu"]
    um = hier.per_pair["t1::miiv"] | hier.per_pair["t2::miiv"]
    assert hier.per_dataset["eicu"] == ue
    assert hier.per_dataset["miiv"] == um


def test_demo_data_tree_exists():
    """Anchor tests to repo demo_data layout (attrition only; no parquets required)."""
    demo = _YAIB_ROOT / "demo_data"
    assert demo.is_dir()
    for task, sub in [
        ("aki", "eicu_demo"),
        ("aki", "mimic_demo"),
        ("mortality24", "eicu_demo"),
    ]:
        attr = demo / task / sub / "attrition.csv"
        assert attr.is_file(), f"missing {attr}"
