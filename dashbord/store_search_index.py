"""
store_search_index.py
----------------------
Dedicated, in-memory autocomplete index for the Store Search experience
on the app landing page (Task 1).

Design goals:
  * Google-style live suggestions while typing (no Enter required).
  * Full Hebrew support (NFKC normalization + diacritic stripping +
    final-letter folding so "סופרר"/"דהנ" behave like "סופר"/"דהן").
  * Partial-word / substring matching.
  * Lightweight fuzzy matching (bounded Levenshtein) so a typo like
    "סופרר" still finds "סופר".
  * Built ONCE and reused. Queries never rescan the raw store list —
    they walk a prebuilt normalized map — so it stays fast from 10 to
    10,000 stores.

The index is intentionally free of any Streamlit / app.py dependency so
it can be unit-tested and reused (e.g. a future REST search endpoint).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# Hebrew final letters -> base letters, so a query typed with (or without)
# a final form still matches. Folded on both sides of every comparison.
_FINALS = {
    "ך": "כ",
    "ם": "מ",
    "ן": "נ",
    "ף": "פ",
    "ץ": "צ",
}


def normalize(text: Optional[str]) -> str:
    """Canonical form used for every index key and every query.

    NFKC-normalize, lowercase, strip combining marks (niqqud/diacritics),
    fold Hebrew final letters, and collapse whitespace. Safe on None.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    out = []
    for ch in text:
        if unicodedata.category(ch) == "Mn":  # combining mark / diacritic
            continue
        out.append(_FINALS.get(ch, ch))
    return " ".join("".join(out).split())


def _levenshtein_within(a: str, b: str, max_dist: int) -> Optional[int]:
    """Bounded edit distance. Returns the distance if <= max_dist, else None.

    Early-exits when the whole row is already past the bound, so it stays
    cheap even across a large candidate set.
    """
    if abs(len(a) - len(b)) > max_dist:
        return None
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(val)
            if val < best:
                best = val
        if best > max_dist:
            return None
        prev = cur
    return prev[-1] if prev[-1] <= max_dist else None


@dataclass(frozen=True)
class SearchRecord:
    """One indexable store, decoupled from app.py's Store dataclass."""

    store_id: str
    name: str                      # primary display name (shown in the suggestion)
    logo_uri: str = ""             # data: URI or URL for the suggestion thumbnail
    branch: str = ""               # optional branch label
    city: str = ""                 # optional city label
    aliases: Tuple[str, ...] = ()  # extra searchable names (e.g. Hebrew name)


@dataclass(frozen=True)
class Suggestion:
    store_id: str
    name: str
    logo_uri: str
    branch: str
    city: str
    score: float  # lower is better (0 = exact / prefix)


class StoreSearchIndex:
    """Prebuilt, reusable autocomplete index over a set of stores."""

    def __init__(self, records: Iterable[SearchRecord]) -> None:
        self._records: Dict[str, SearchRecord] = {}
        # normalized searchable string -> list of store_ids
        self._by_norm: Dict[str, List[str]] = {}
        for rec in records:
            self._add(rec)

    def _add(self, rec: SearchRecord) -> None:
        self._records[rec.store_id] = rec
        searchable = [rec.name, rec.branch, rec.city, *rec.aliases]
        for raw in searchable:
            norm = normalize(raw)
            if not norm:
                continue
            ids = self._by_norm.setdefault(norm, [])
            if rec.store_id not in ids:
                ids.append(rec.store_id)

    def __len__(self) -> int:
        return len(self._records)

    def suggest(self, query: str, limit: int = 8) -> List[Suggestion]:
        """Return ranked suggestions for a (possibly partial/typo'd) query."""
        q = normalize(query)
        if not q:
            return []

        # store_id -> best (lowest) score seen for it
        scored: Dict[str, float] = {}

        def consider(store_id: str, score: float) -> None:
            if store_id not in scored or score < scored[store_id]:
                scored[store_id] = score

        # 1) Exact / prefix / substring matches against full searchable strings.
        for norm, ids in self._by_norm.items():
            score: Optional[float] = None
            if norm == q:
                score = 0.0
            elif norm.startswith(q):
                score = 0.1
            elif any(tok.startswith(q) for tok in norm.split(" ")):
                score = 0.2
            elif q in norm:
                score = 0.4
            if score is not None:
                for sid in ids:
                    consider(sid, score)

        # 2) Fuzzy fallback — only when the query is long enough to be a
        #    meaningful typo, and bounded so it never explodes.
        if len(q) >= 3:
            max_dist = 1 if len(q) <= 5 else 2
            for norm, ids in self._by_norm.items():
                for token in norm.split(" ") + [norm]:
                    dist = _levenshtein_within(q, token, max_dist)
                    if dist is not None:
                        fuzzy_score = 0.5 + 0.15 * dist
                        for sid in ids:
                            consider(sid, fuzzy_score)
                        break

        ranked: List[Tuple[float, str]] = sorted(
            ((score, sid) for sid, score in scored.items()),
            key=lambda pair: (pair[0], self._records[pair[1]].name),
        )

        results: List[Suggestion] = []
        for score, sid in ranked[:limit]:
            rec = self._records[sid]
            results.append(
                Suggestion(
                    store_id=rec.store_id,
                    name=rec.name,
                    logo_uri=rec.logo_uri,
                    branch=rec.branch,
                    city=rec.city,
                    score=score,
                )
            )
        return results
