"""Build deterministic train/val/test splits + labels.json.

This replaces ``archive/notebooks/split_dataset.ipynb`` (original: "01 -
Simple dataset split"). It is a **corrected** version — the original split
by a plain seeded shuffle of individual videos, which is unsafe: it does not
prevent the same identity (e.g. FaceForensics++ target/source id) from
appearing in both train and a val/test split. See ``src/data/identity.py``
and ``src/eval/leakage_check.py`` for the full rationale.

This script now groups videos by identity *before* splitting, so an entire
identity's videos are assigned to exactly one split. It's a greedy
size-balanced bin-packing over identity groups, not exact stratified
sampling — perfect stratification under a hard grouping constraint is a
bin-packing problem with no guaranteed-exact solution, so this trades a small
amount of class-balance precision for a hard leakage guarantee, and reports
the achieved balance so any drift is visible rather than silently accepted.

Expected input layout (see HANDOFF.md "Dataset Required" — this exact layout
was inferred from ``archive/notebooks/split_dataset.ipynb`` and has NOT been
independently verified against a real dataset in this environment):

    <dataset_dir>/DFDC_REAL_Face_only_data/<video_stem>/...
    <dataset_dir>/DFDC_FAKE_Face_only_data/<video_stem>/...
    <dataset_dir>/FF_Face_only_data/<video_stem>/...
    <dataset_dir>/FF_Face_only_data/metadata.csv        (optional)
    <dataset_dir>/Celeb_real_face_only/<video_stem>/...  (optional, held out)
    <dataset_dir>/Celeb_fake_face_only/<video_stem>/...  (optional, held out)

Usage:

    python -m src.data.build_splits --check-leakage

Do NOT run this without the dataset in place — see HANDOFF.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src import config
from src.data.identity import group_stems_by_identity

logger = logging.getLogger(__name__)

DFDC_REAL_DIRNAME = "DFDC_REAL_Face_only_data"
DFDC_FAKE_DIRNAME = "DFDC_FAKE_Face_only_data"
FFPP_DIRNAME = "FF_Face_only_data"
FFPP_METADATA_FILENAME = "metadata.csv"
CELEB_REAL_DIRNAME = "Celeb_real_face_only"
CELEB_FAKE_DIRNAME = "Celeb_fake_face_only"

TRAIN_FRAC_DEFAULT = 0.80
VAL_FRAC_DEFAULT = 0.15
RESERVE_COUNT_DEFAULT = 200
SEED_DEFAULT = 42


@dataclass
class DatasetEntry:
    path: str
    stem: str
    label: int  # 0 = real, 1 = fake
    source: str


def list_videos(folder: Path) -> list[Path]:
    """One entry per subfolder if the folder contains per-video subfolders
    (the expected "face-only" preprocessed layout); otherwise one entry per
    file directly inside it."""
    if not folder.exists():
        return []
    subdirs = [p for p in folder.iterdir() if p.is_dir()]
    if subdirs:
        return subdirs
    return [p for p in folder.glob("*") if p.is_file()]


def _parse_ffpp_metadata(metadata_csv: Path) -> dict[str, int]:
    """Best-effort parse of an FF++ metadata.csv into {video_stem: label}.

    Mirrors the heuristic from the original notebook: prefer explicit
    'video'/'label' columns; otherwise guess the label column by name
    (containing 'label', 'fake', 'class', 'manipulated').
    """
    import pandas as pd

    mapping: dict[str, int] = {}
    try:
        df = pd.read_csv(metadata_csv)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse FF++ metadata.csv at %s: %s", metadata_csv, exc)
        return mapping

    if "video" in df.columns and "label" in df.columns:
        for _, row in df.iterrows():
            mapping[str(Path(row["video"]).stem)] = int(row["label"])
        return mapping

    cols = list(df.columns)
    if not cols:
        return mapping
    key_col = cols[0]
    label_col = next(
        (c for c in cols[1:] if any(x in c.lower() for x in ("label", "fake", "class", "manipulated"))),
        cols[1] if len(cols) > 1 else None,
    )
    if label_col is None:
        return mapping
    for _, row in df.iterrows():
        truthy = str(row[label_col]).strip().lower() in ("1", "true", "fake", "t", "y", "yes")
        mapping[str(Path(row[key_col]).stem)] = 1 if truthy else 0
    return mapping


def collect_entries(dataset_dir: Path) -> tuple[list[DatasetEntry], list[DatasetEntry]]:
    """Returns (trainable_entries, celeb_holdout_entries).

    Celeb-DF is collected but kept out of the trainable pool, matching the
    original notebook's intent ("optional: not used for training") — it's
    written to its own file so it's available as an external held-out
    evaluation set later if desired.
    """
    entries: list[DatasetEntry] = []

    for p in list_videos(dataset_dir / DFDC_REAL_DIRNAME):
        entries.append(DatasetEntry(path=str(p), stem=p.stem, label=0, source="dfdc_real"))
    for p in list_videos(dataset_dir / DFDC_FAKE_DIRNAME):
        entries.append(DatasetEntry(path=str(p), stem=p.stem, label=1, source="dfdc_fake"))

    ffpp_dir = dataset_dir / FFPP_DIRNAME
    ffpp_meta = _parse_ffpp_metadata(ffpp_dir / FFPP_METADATA_FILENAME) if (ffpp_dir / FFPP_METADATA_FILENAME).exists() else {}
    for p in list_videos(ffpp_dir):
        stem = p.stem
        if stem in ffpp_meta:
            label = ffpp_meta[stem]
        else:
            parent = str(p.parent).lower()
            if "fake" in parent:
                label = 1
            elif "real" in parent:
                label = 0
            else:
                logger.warning("Skipping FF++ entry with no recoverable label: %s", p)
                continue
        entries.append(DatasetEntry(path=str(p), stem=stem, label=int(label), source="ffpp"))

    celeb_entries: list[DatasetEntry] = []
    for p in list_videos(dataset_dir / CELEB_REAL_DIRNAME):
        celeb_entries.append(DatasetEntry(path=str(p), stem=p.stem, label=0, source="celeb_real"))
    for p in list_videos(dataset_dir / CELEB_FAKE_DIRNAME):
        celeb_entries.append(DatasetEntry(path=str(p), stem=p.stem, label=1, source="celeb_fake"))

    return entries, celeb_entries


@dataclass
class _GroupWithEntries:
    group_id: str
    confidence: str
    entries: list[DatasetEntry] = field(default_factory=list)


def assign_groups_to_splits(
    entries: list[DatasetEntry],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    reserve_count: int,
) -> dict[str, list[DatasetEntry]]:
    """Group entries by identity, then greedily assign whole groups to
    train/val/test/reserved to hit target size fractions (deficit-based
    bin-balancing — largest groups placed first, each into whichever split
    is currently furthest below its target)."""
    entries_by_stem: dict[str, list[DatasetEntry]] = {}
    for e in entries:
        entries_by_stem.setdefault(e.stem, []).append(e)

    id_groups = group_stems_by_identity(list(entries_by_stem.keys()))
    groups = [
        _GroupWithEntries(
            group_id=g.group_id,
            confidence=g.confidence,
            entries=[e for stem in g.stems for e in entries_by_stem[stem]],
        )
        for g in id_groups.values()
    ]

    rng = random.Random(seed)
    order = sorted(groups, key=lambda g: g.group_id)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(g.entries))  # largest first; stable, so shuffle above breaks ties

    total = sum(len(g.entries) for g in order)
    remaining_after_reserve = max(total - reserve_count, 0)
    targets = {
        "reserved": reserve_count,
        "train": int(remaining_after_reserve * train_frac),
        "val": int(remaining_after_reserve * val_frac),
    }
    targets["test"] = remaining_after_reserve - targets["train"] - targets["val"]

    counts = dict.fromkeys(targets, 0)
    assigned: dict[str, list[DatasetEntry]] = {k: [] for k in targets}

    for g in order:
        deficits = {k: targets[k] - counts[k] for k in targets}
        best_split = max(deficits.items(), key=lambda kv: (kv[1], kv[0]))[0]
        assigned[best_split].extend(g.entries)
        counts[best_split] += len(g.entries)

    return assigned


def write_outputs(assigned: dict[str, list[DatasetEntry]], celeb_entries: list[DatasetEntry], out_dir: Path, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_list(items: list[DatasetEntry], filename: str) -> None:
        with open(out_dir / filename, "w", encoding="utf-8") as f:
            for it in items:
                f.write(it.path + "\n")

    write_list(assigned["train"], "train.txt")
    write_list(assigned["val"], "val.txt")
    write_list(assigned["test"], "test_internal.txt")
    write_list(assigned["reserved"], "reserved_200.txt")
    write_list(celeb_entries, "celeb_holdout.txt")

    labels: dict[str, int] = {}
    for it in assigned["train"] + assigned["val"] + assigned["test"] + assigned["reserved"] + celeb_entries:
        key = it.stem
        if key in labels:
            key = f"{it.stem}__{it.source}"
        labels[key] = it.label
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    manifest = {
        "seed": seed,
        "splits": {
            name: {
                "count": len(items),
                "label_counts": dict(Counter(it.label for it in items)),
            }
            for name, items in assigned.items()
        },
        "celeb_holdout": {
            "count": len(celeb_entries),
            "label_counts": dict(Counter(it.label for it in celeb_entries)),
            "note": "Collected but not used for training, matching the original notebook's intent.",
        },
        "leakage_prevention": (
            "Splits are assigned per identity-group (src/data/identity.py), not per video. "
            "DFDC entries are opaque single-stem identities (no metadata.json preserved) — "
            "leakage prevention only covers FF++ target/source id pairs. Run "
            "`python -m src.eval.leakage_check` to verify."
        ),
    }
    (out_dir / "data_manifest.yaml").write_text(yaml.dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC_DEFAULT)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC_DEFAULT)
    parser.add_argument("--reserve-count", type=int, default=RESERVE_COUNT_DEFAULT)
    parser.add_argument(
        "--check-leakage",
        action="store_true",
        help="Run src.eval.leakage_check on the freshly written splits and fail if any identity leaks across splits.",
    )
    args = parser.parse_args(argv)

    if not args.dataset_dir.exists():
        logger.error(
            "Dataset directory not found: %s. This script requires the real dataset "
            "to be present — see HANDOFF.md 'Dataset Required'. Not running further.",
            args.dataset_dir,
        )
        return 2

    entries, celeb_entries = collect_entries(args.dataset_dir)
    if not entries:
        logger.error("No entries collected from %s — check folder names against HANDOFF.md.", args.dataset_dir)
        return 2

    logger.info("Collected %d trainable entries, %d Celeb-DF holdout entries", len(entries), len(celeb_entries))

    assigned = assign_groups_to_splits(
        entries,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        reserve_count=args.reserve_count,
    )
    write_outputs(assigned, celeb_entries, args.out_dir, seed=args.seed)

    for name, items in assigned.items():
        logger.info("%s: %d videos, label_counts=%s", name, len(items), dict(Counter(it.label for it in items)))

    if args.check_leakage:
        from src.eval.leakage_check import check_leakage, stem_to_split_map

        stem_to_split = stem_to_split_map(
            {
                "train": args.out_dir / "train.txt",
                "val": args.out_dir / "val.txt",
                "test": args.out_dir / "test_internal.txt",
            }
        )
        report = check_leakage(stem_to_split)
        (args.out_dir / "leakage_report.json").write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
        if report.has_leakage:
            logger.error(
                "Leakage check FAILED: %d identity groups span multiple splits. See %s",
                len(report.leaking_groups),
                args.out_dir / "leakage_report.json",
            )
            return 1
        logger.info("Leakage check passed: 0 identity groups span multiple splits (FF++ pairs only; see note in report).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
