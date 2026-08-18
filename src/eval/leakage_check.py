"""Identity-leakage checker for the train/val/test splits.

Standalone script — run it once real data is present (see HANDOFF.md,
"Exact Run Order"). It is deliberately NOT executed as part of writing this
codebase, since no dataset is available in this environment.

Why this exists: the original codebase's only leak check
(``notebooks/check_embeddings.ipynb``) verified there was no *stem* overlap
across ``embeddings/{train,val,test}/*.npy``. That catches duplicate files,
not shared identities. FaceForensics++ stems encode two identities as
``<target_id>_<source_id>``; two different, non-overlapping stems (e.g.
``005_010`` in train and ``005_044`` in test) can still share identity 005.
See ``src/data/identity.py`` for the grouping logic and its documented
limitation: DFDC entries have no recoverable identity mapping in this
project (no ``metadata.json`` was ever preserved), so they are treated as
single-stem opaque identities and this checker **cannot** detect DFDC-side
leakage — only FF++-side.

Usage (once ``data/labels.json`` + split files exist):

    python -m src.eval.leakage_check \\
        --labels data/labels.json \\
        --train data/train.txt --val data/val.txt --test data/test_internal.txt \\
        --out results/leakage_report.json

Exit code is non-zero if any identity group spans more than one split, so
this can be wired into CI/the pipeline script as a hard gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.data.identity import group_stems_by_identity


@dataclass
class LeakageFinding:
    group_id: str
    stems: list[str]
    splits: dict[str, int]  # split_name -> count of stems from this group in that split
    confidence: str


@dataclass
class LeakageReport:
    total_stems: int
    total_identity_groups: int
    ffpp_pair_groups: int
    dfdc_pair_groups: int
    opaque_groups: int
    leaking_groups: list[LeakageFinding] = field(default_factory=list)
    note: str = (
        "DFDC-side leakage is checked via preserved metadata.json "
        "(fake->original real-video mapping), when available — see "
        "src/data/identity.py. Coverage is per-fake-video source pairing "
        "only; two REAL videos of the same person with no fake derived "
        "from either are still treated as separate identities, since DFDC "
        "metadata does not expose an actor/person ID beyond that."
    )

    @property
    def has_leakage(self) -> bool:
        return len(self.leaking_groups) > 0

    def to_json(self) -> dict:
        return asdict(self) | {"has_leakage": self.has_leakage}


def stem_to_split_map(split_files: dict[str, Path]) -> dict[str, str]:
    """Build stem -> split name from the plain path-list files written by
    ``src/data/build_splits.py`` (one filesystem path per line; the video
    stem is used as the identity key, matching ``labels.json``)."""
    mapping: dict[str, str] = {}
    for split_name, path in split_files.items():
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            stem = Path(line).stem
            mapping[stem] = split_name
    return mapping


def check_leakage(stem_to_split: dict[str, str]) -> LeakageReport:
    """Core check: group all known stems by identity, flag any group whose
    members land in more than one split."""
    groups = group_stems_by_identity(list(stem_to_split.keys()))

    leaking: list[LeakageFinding] = []
    ffpp_pair_groups = 0
    dfdc_pair_groups = 0
    opaque_groups = 0

    for group in groups.values():
        if group.confidence == "ffpp_pair":
            ffpp_pair_groups += 1
        elif group.confidence == "dfdc_pair":
            dfdc_pair_groups += 1
        else:
            opaque_groups += 1

        split_counts: dict[str, int] = {}
        for stem in group.stems:
            split = stem_to_split.get(stem)
            if split is None:
                continue
            split_counts[split] = split_counts.get(split, 0) + 1

        if len(split_counts) > 1:
            leaking.append(
                LeakageFinding(
                    group_id=group.group_id,
                    stems=group.stems,
                    splits=split_counts,
                    confidence=group.confidence,
                )
            )

    return LeakageReport(
        total_stems=len(stem_to_split),
        total_identity_groups=len(groups),
        ffpp_pair_groups=ffpp_pair_groups,
        dfdc_pair_groups=dfdc_pair_groups,
        opaque_groups=opaque_groups,
        leaking_groups=sorted(leaking, key=lambda f: f.group_id),
    )


def _print_human_summary(report: LeakageReport) -> None:
    print(f"Total stems checked:      {report.total_stems}")
    print(f"Total identity groups:    {report.total_identity_groups}")
    print(f"  - FF++ resolved pairs:  {report.ffpp_pair_groups}")
    print(f"  - Opaque (unresolved):  {report.opaque_groups}")
    print(f"Leaking identity groups:  {len(report.leaking_groups)}")
    if report.leaking_groups:
        print("\nFirst 20 leaking groups:")
        for finding in report.leaking_groups[:20]:
            print(f"  [{finding.confidence}] {finding.group_id}: splits={finding.splits} stems={finding.stems[:6]}{'...' if len(finding.stems) > 6 else ''}")
    print(f"\nNOTE: {report.note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True, help="data/train.txt")
    parser.add_argument("--val", type=Path, required=True, help="data/val.txt")
    parser.add_argument("--test", type=Path, required=True, help="data/test_internal.txt")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write the JSON report")
    args = parser.parse_args(argv)

    stem_to_split = stem_to_split_map({"train": args.train, "val": args.val, "test": args.test})
    if not stem_to_split:
        print(
            "No stems found in the given split files — dataset is not present "
            "in this environment (expected; see HANDOFF.md).",
            file=sys.stderr,
        )
        return 2

    report = check_leakage(stem_to_split)
    _print_human_summary(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
        print(f"\nWrote report to {args.out}")

    return 1 if report.has_leakage else 0


if __name__ == "__main__":
    raise SystemExit(main())
