
"""Identity extraction and grouping, shared by split generation and the
leakage checker (``src/eval/leakage_check.py``).

... (original docstring content preserved above this point in the real file)

UPDATE: DFDC identity resolution via preserved metadata.json.

Original limitation (now partially closed): DFDC entries were previously
treated as opaque single-stem identities because no metadata.json survived
into this project. That gap has been closed for a specific dataset pull
(dfdc-10, 10 parts, 19,909 entries) whose per-part metadata.json files were
preserved and combined into dfdc_combined_metadata.json, which maps each
fake video stem to its `original` (source real video) stem. This module now
loads that mapping, when present, and unions a DFDC fake stem with its real
source stem into the same identity group -- closing DFDC-side leakage
detection for any project run against that specific combined metadata file.

This does NOT resolve DFDC real-video-to-actor-identity links beyond direct
fake->source pairs (DFDC's metadata does not expose a separate actor/person
ID beyond the video-level real/fake/original relationship) -- two REAL
videos of the same person with no fake derived from either would still be
treated as separate opaque identities. This is a real, remaining limitation,
not swept under the rug.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_FFPP_STEM_RE = re.compile(r"^(\\d+)_(\\d+)$")

_DFDC_METADATA_ENV_VAR = "VERIFACE_DFDC_METADATA_PATH"
_DFDC_METADATA_DEFAULT = Path("/workspace/veriface-dataset/dfdc_combined_metadata.json")


def _load_dfdc_metadata() -> dict[str, dict]:
    """Load the combined DFDC metadata (stem -> {label, original_stem, part}),
    if present. Returns {} if not found -- callers must treat DFDC stems as
    opaque in that case, exactly as before this update."""
    path_str = os.environ.get(_DFDC_METADATA_ENV_VAR)
    path = Path(path_str) if path_str else _DFDC_METADATA_DEFAULT
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_DFDC_METADATA_CACHE: dict[str, dict] | None = None


def _dfdc_metadata() -> dict[str, dict]:
    global _DFDC_METADATA_CACHE
    if _DFDC_METADATA_CACHE is None:
        _DFDC_METADATA_CACHE = _load_dfdc_metadata()
    return _DFDC_METADATA_CACHE


@dataclass(frozen=True)
class IdentityTokens:
    stem: str
    tokens: tuple[str, ...]
    confidence: str  # "ffpp_pair" | "dfdc_pair" | "opaque"


def parse_identity_tokens(stem: str) -> IdentityTokens:
    """Parse identity tokens out of a video stem.

    - FF++-style ``NNN_MMM`` stems yield two tokens (unchanged from before).
    - DFDC stems with a resolved `original_stem` in the combined metadata
      yield two tokens: the stem itself and its source real video's stem,
      both namespaced "dfdc:" -- so a DFDC hash can never collide with an
      FF++ numeric id or an unrelated DFDC hash.
    - Anything else remains a single opaque identity token, exactly as
      before.
    """
    base = stem.split("__", 1)[0]

    match = _FFPP_STEM_RE.match(base)
    if match:
        target_id, source_id = match.groups()
        return IdentityTokens(
            stem=stem,
            tokens=(f"ffpp:{target_id}", f"ffpp:{source_id}"),
            confidence="ffpp_pair",
        )

    dfdc_meta = _dfdc_metadata()
    entry = dfdc_meta.get(base) or dfdc_meta.get(stem)
    if entry and "original_stem" in entry:
        return IdentityTokens(
            stem=stem,
            tokens=(f"dfdc:{base}", f"dfdc:{entry['original_stem']}"),
            confidence="dfdc_pair",
        )
    if entry:
        # Real DFDC video with no fake derived from it (or not the source
        # of one we know about) -- still resolvable as its own token, just
        # not linked to anything else yet.
        return IdentityTokens(stem=stem, tokens=(f"dfdc:{base}",), confidence="opaque")

    return IdentityTokens(stem=stem, tokens=(f"opaque:{base}",), confidence="opaque")


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


@dataclass
class IdentityGroup:
    group_id: str
    stems: list[str] = field(default_factory=list)
    confidence: str = "opaque"


def group_stems_by_identity(stems: list[str]) -> dict[str, IdentityGroup]:
    uf = _UnionFind()
    parsed = [parse_identity_tokens(s) for s in stems]

    for p in parsed:
        for tok in p.tokens:
            uf.union(p.tokens[0], tok)

    members: dict[str, list[IdentityTokens]] = defaultdict(list)
    for p in parsed:
        root = uf.find(p.tokens[0])
        members[root].append(p)

    groups: dict[str, IdentityGroup] = {}
    for root, items in members.items():
        stems_in_group = sorted(i.stem for i in items)
        group_id = stems_in_group[0]
        confidences = {i.confidence for i in items}
        if "ffpp_pair" in confidences:
            confidence = "ffpp_pair"
        elif "dfdc_pair" in confidences:
            confidence = "dfdc_pair"
        else:
            confidence = "opaque"
        groups[group_id] = IdentityGroup(group_id=group_id, stems=stems_in_group, confidence=confidence)

    return groups
