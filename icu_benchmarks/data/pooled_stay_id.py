"""Stay_id suffixing for pooled multi-corpus merges (matches Bachelor merge_pooled_corpora.py).

``merge_pooled_corpora`` uses ``int(str(int(x)) + suffix)`` with ``suffix = str(idx + 1) * 4``
for each ``--corpus-dirs`` index ``idx`` (0-based). Tri default order eicu, hirid, miiv →
1111, 2222, 3333.
"""

from __future__ import annotations


def pooled_suffix_digits(corpus_index_zero_based: int) -> str:
    """Four repeated digits: 1111, 2222, … for index 0, 1, …"""
    if corpus_index_zero_based < 0:
        raise ValueError("corpus_index_zero_based must be >= 0")
    return str(corpus_index_zero_based + 1) * 4


def apply_pooled_stay_id_suffix(stay_id: int, corpus_index_zero_based: int) -> int:
    """Map leaf corpus stay_id to merged pooled stay_id (same formula as merge_pooled_corpora)."""
    suf = pooled_suffix_digits(corpus_index_zero_based)
    return int(str(int(stay_id)) + suf)


def pooled_index_map_from_merge_order(dataset_slugs_in_merge_order: list[str]) -> dict[str, int]:
    """Slug → 0-based corpus index, matching merge_pooled_corpora --corpus-dirs order."""
    out: dict[str, int] = {}
    for i, slug in enumerate(dataset_slugs_in_merge_order):
        s = slug.strip()
        if not s:
            continue
        if s in out:
            raise ValueError(f"Duplicate dataset slug in pooled merge order: {s!r}")
        out[s] = i
    if not out:
        raise ValueError("pooled merge order is empty")
    return out
