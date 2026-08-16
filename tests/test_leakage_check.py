"""Tests for identity grouping + leakage detection, using synthetic
FF++-style and DFDC-style stems only — no real dataset required, per
HANDOFF.md constraints on this environment.
"""

from __future__ import annotations

from src.data.identity import group_stems_by_identity, parse_identity_tokens
from src.eval.leakage_check import check_leakage


def test_parse_identity_tokens_ffpp_pair():
    tokens = parse_identity_tokens("005_010")
    assert tokens.confidence == "ffpp_pair"
    assert set(tokens.tokens) == {"ffpp:005", "ffpp:010"}


def test_parse_identity_tokens_opaque_dfdc_style():
    tokens = parse_identity_tokens("837")
    assert tokens.confidence == "opaque"
    assert tokens.tokens == ("opaque:837",)


def test_parse_identity_tokens_strips_labels_json_dedup_suffix():
    """labels.json disambiguates duplicate stems with a __<source> suffix
    (see src/data/build_splits.py) — must be stripped before parsing."""
    tokens = parse_identity_tokens("005_010__ffpp")
    assert tokens.confidence == "ffpp_pair"
    assert set(tokens.tokens) == {"ffpp:005", "ffpp:010"}


def test_group_stems_by_identity_transitively_links_shared_ids():
    """005_010 and 005_044 share target identity 005 and must land in the
    same group even though their stems are entirely different strings —
    exactly the leak the original stem-only overlap check
    (archive/notebooks/check_embeddings.ipynb) could not catch."""
    groups = group_stems_by_identity(["005_010", "005_044", "099_001"])
    group_ids = {stem: g.group_id for g in groups.values() for stem in g.stems}
    assert group_ids["005_010"] == group_ids["005_044"]
    assert group_ids["099_001"] != group_ids["005_010"]


def test_group_stems_by_identity_opaque_stems_stay_singleton():
    groups = group_stems_by_identity(["111", "222", "333"])
    assert len(groups) == 3
    for g in groups.values():
        assert len(g.stems) == 1
        assert g.confidence == "opaque"


def test_check_leakage_detects_ffpp_identity_split_across_train_and_test():
    stem_to_split = {
        "005_010": "train",
        "005_044": "test",  # same target identity 005 as above, but in test
        "099_001": "train",
        "099_002": "val",  # different identity family, no leak
        "100_200": "train",
        "100_300": "train",  # same group, but both in train — not a leak
    }
    report = check_leakage(stem_to_split)

    assert report.has_leakage
    leaking_ids = {finding.group_id for finding in report.leaking_groups}
    # 005_010/005_044 share id 005 -> group_id is the min stem, "005_010"
    assert "005_010" in leaking_ids
    # 100_200/100_300 both in train -> not leaking
    assert "100_200" not in leaking_ids


def test_check_leakage_clean_splits_report_no_leakage():
    stem_to_split = {
        "005_010": "train",
        "099_001": "val",
        "150_151": "test",
        "837": "train",
        "838": "val",
    }
    report = check_leakage(stem_to_split)
    assert not report.has_leakage
    assert report.total_stems == 5


def test_check_leakage_report_flags_dfdc_limitation_in_note():
    report = check_leakage({"1": "train"})
    assert "DFDC" in report.note
