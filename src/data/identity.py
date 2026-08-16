"""Identity extraction and grouping, shared by split generation and the
leakage checker (``src/eval/leakage_check.py``).

Background (see KNOWLEDGE_BASE.md / HANDOFF.md "Dataset Required" for the
full audit): the original pipeline split videos by *stem* only
(``notebooks/check_embeddings.ipynb`` even had a "no stem overlap between
splits" sanity check), which does **not** catch identity leakage. FaceForensics++
manipulated-video stems encode two identities as ``<target_id>_<source_id>``
(e.g. ``005_010.mp4`` — identity 005 manipulated using identity 010's
face/expression). If identity 005 or 010 appears in *any* video assigned to
train, and again in *any* video assigned to val/test, that's a leak: the
model can learn to recognize the person rather than the manipulation
artifact.

Known, explicit limitation (do not paper over this): DFDC entries in this
project were reindexed to bare integers (e.g. ``837.npy``) with no
preserved link back to DFDC's own ``metadata.json`` (which maps each fake to
its real source video and to an actor id). A DFDC-style stem is therefore
treated as an *opaque, low-confidence* identity — its own stem is its only
known "identity token", so this checker cannot detect DFDC-side identity
leakage at all. It can only detect it for FF++-style stems. This is flagged
in every report this module produces; do not assume a "no leakage" result
covers DFDC entries.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

_FFPP_STEM_RE = re.compile(r"^(\d+)_(\d+)$")


@dataclass(frozen=True)
class IdentityTokens:
    """Identity tokens parsed from one video stem."""

    stem: str
    tokens: tuple[str, ...]
    confidence: str  # "ffpp_pair" | "opaque"


def parse_identity_tokens(stem: str) -> IdentityTokens:
    """Parse identity tokens out of a video stem.

    ``labels.json`` disambiguates duplicate stems with a ``__<source>``
    suffix (see ``src/data/build_splits.py``); strip that first.

    - FF++-style ``NNN_MMM`` stems yield two tokens, each namespaced by
      ``"ffpp:"`` so a numeric id can never collide with an unrelated
      DFDC integer stem that happens to share the same digits.
    - Anything else (DFDC reindexed integers, Celeb-DF, unrecognized) is
      treated as its own single, opaque identity token — i.e. we assume
      no sharing unless proven otherwise, which is the honest thing to do
      given no source mapping survives for those entries.
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
    return IdentityTokens(stem=stem, tokens=(f"opaque:{base}",), confidence="opaque")


class _UnionFind:
    """Minimal union-find (disjoint set) over arbitrary hashable tokens."""

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
    """A connected component of stems that share at least one identity token."""

    group_id: str
    stems: list[str] = field(default_factory=list)
    confidence: str = "opaque"  # "ffpp_pair" if any member had a resolved id pair


def group_stems_by_identity(stems: list[str]) -> dict[str, IdentityGroup]:
    """Group video stems into identity-connected components.

    Two stems land in the same group if they share any identity token (e.g.
    both involve FF++ identity ``005``, as either target or source). Returns
    a mapping of ``group_id -> IdentityGroup``. ``group_id`` is deterministic
    (min stem in the group) so results are stable across runs for the same
    input set.
    """
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
        confidence = "ffpp_pair" if any(i.confidence == "ffpp_pair" for i in items) else "opaque"
        groups[group_id] = IdentityGroup(group_id=group_id, stems=stems_in_group, confidence=confidence)

    return groups
