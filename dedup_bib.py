#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dedup_bib.py - Deduplicate citations across multiple collaborator BibTeX files.

Workflow:
1) Read all input .bib files.
2) Cluster duplicates by strong IDs (DOI/arXiv/OpenReview).
3) Bridge no-ID entries with fuzzy signature (title + first author + year).
4) Pick canonical entry per cluster, merge missing fields, keep one citekey.
5) Write merged deduplicated bib + mapping/report artifacts.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import bibtexparser
    from bibtexparser.bwriter import BibTexWriter
except ImportError:
    bibtexparser = None
    BibTexWriter = None

DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
ARXIV_NEW_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b", re.IGNORECASE)
ARXIV_OLD_RE = re.compile(r"\b([a-z\-]+(\.[A-Z]{2})?/\d{7})(v\d+)?\b", re.IGNORECASE)
OPENREVIEW_RE = re.compile(r"openreview\.net/(?:forum|pdf)\?id=([A-Za-z0-9_\-]+)")
ARXIV_ONLY_FIELDS = {"eprint", "eprinttype", "archiveprefix", "primaryclass"}


def ensure_dependencies() -> None:
    if bibtexparser is None or BibTexWriter is None:
        raise SystemExit("Missing dependency: bibtexparser. Install it first, e.g. `pip install bibtexparser`.")


def norm_space(s: str) -> str:
    return " ".join((s or "").replace("\n", " ").split()).strip()


def strip_braces(s: str) -> str:
    return (s or "").replace("{", "").replace("}", "")


def normalize_title_for_match(title: str) -> str:
    t = strip_braces(norm_space(title)).lower()
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def sanitize_doi(doi: str) -> str:
    d = norm_space(doi)
    d = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", d, flags=re.IGNORECASE)
    prev = None
    while d and d != prev:
        prev = d
        d = d.strip()
        d = d.rstrip(".,;")
        d = d.strip("<>\"'")
    for open_c, close_c in (("(", ")"), ("[", "]"), ("{", "}")):
        while d.startswith(open_c) and d.endswith(close_c):
            inner = d[1:-1].strip()
            if inner.lower().startswith("10."):
                d = inner
                continue
            break
    for open_c, close_c in (("(", ")"), ("[", "]"), ("{", "}")):
        while d.endswith(close_c) and d.count(close_c) > d.count(open_c):
            d = d[:-1]
        while d.startswith(open_c) and d.count(open_c) > d.count(close_c):
            d = d[1:]
    return d.strip().lower()


def extract_doi(s: str) -> Optional[str]:
    if not s:
        return None
    m = DOI_RE.search(s)
    if not m:
        return None
    doi = sanitize_doi(m.group(1))
    return doi or None


def extract_arxiv_id(s: str) -> Optional[str]:
    if not s:
        return None
    m = ARXIV_NEW_RE.search(s)
    if m:
        return (m.group(1) + (m.group(2) or "")).lower()
    m = ARXIV_OLD_RE.search(s)
    if m:
        return (m.group(1) + (m.group(3) or "")).lower()
    return None


def arxiv_base_id(s: str) -> Optional[str]:
    arx = extract_arxiv_id(s)
    if not arx:
        return None
    return re.sub(r"v\d+$", "", arx.lower())


def extract_openreview_id(s: str) -> Optional[str]:
    if not s:
        return None
    m = OPENREVIEW_RE.search(s)
    return m.group(1) if m else None


def author_lastnames(author_field: str, k: int = 1) -> List[str]:
    if not author_field:
        return []
    parts = [p.strip() for p in author_field.split(" and ") if p.strip()]
    out: List[str] = []
    for p in parts[:k]:
        if "," in p:
            ln = p.split(",")[0].strip()
        else:
            ln = p.split()[-1].strip()
        if ln:
            out.append(ln.lower())
    return out


def extract_year(year_value: str) -> str:
    m = re.search(r"(?:19|20)\d{2}", str(year_value or ""))
    return m.group(0) if m else ""


def has_venue_info(entry: Dict[str, str]) -> bool:
    return bool(norm_space(entry.get("booktitle") or entry.get("journal") or ""))


def extract_ids_from_entry(entry: Dict[str, str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    blob_fields = ["doi", "url", "note", "howpublished", "eprint", "journal", "booktitle"]
    blob = " ".join([entry.get(f, "") for f in blob_fields if entry.get(f)])
    doi = extract_doi(blob)
    arxiv = arxiv_base_id(blob)
    openreview = extract_openreview_id(blob)
    return doi, arxiv, openreview


def fuzzy_signature(entry: Dict[str, str]) -> Optional[str]:
    title = normalize_title_for_match(entry.get("title", ""))
    if not title:
        return None
    ln = author_lastnames(entry.get("author", ""), k=1)
    first_ln = ln[0] if ln else ""
    year = extract_year(entry.get("year", ""))
    return f"{title}|{first_ln}|{year}"


def entry_completeness_score(entry: Dict[str, str]) -> int:
    score = 0
    doi, arxiv, openreview = extract_ids_from_entry(entry)
    if doi:
        score += 100
    if arxiv or openreview:
        score += 30
    if has_venue_info(entry):
        score += 30
    if entry.get("pages"):
        score += 20
    if entry.get("author"):
        score += 10
    if extract_year(entry.get("year", "")):
        score += 5
    score += len([k for k, v in entry.items() if v])
    return score


def normalize_entry(entry: Dict[str, str]) -> None:
    if entry.get("doi"):
        doi = sanitize_doi(entry.get("doi", ""))
        if doi:
            entry["doi"] = doi
        else:
            entry.pop("doi", None)
    if entry.get("year"):
        y = extract_year(entry.get("year", ""))
        if y:
            entry["year"] = y
    if has_venue_info(entry):
        for f in ARXIV_ONLY_FIELDS:
            entry.pop(f, None)


def merge_entries(rep: Dict[str, str], other: Dict[str, str]) -> None:
    for k, v in other.items():
        if k in ("ID",):
            continue
        if not v:
            continue
        if k not in rep or not rep.get(k):
            rep[k] = v
            continue
        if rep.get(k) == v:
            continue
        # Keep more informative note/url variants without overriding existing stable value.
        if k in ("note", "url"):
            if v not in rep[k]:
                rep[k] = rep[k] + " | " + v

    # If this is a published entry after merge, keep it venue-style.
    if has_venue_info(rep):
        for f in ARXIV_ONLY_FIELDS:
            rep.pop(f, None)


def suggest_citekey(entry: Dict[str, str]) -> str:
    ln = author_lastnames(entry.get("author", ""), k=1)
    author = ln[0] if ln else "ref"
    year = extract_year(entry.get("year", "")) or "noyear"
    title = normalize_title_for_match(entry.get("title", ""))
    word = "work"
    if title:
        for token in title.split():
            if len(token) >= 3:
                word = token
                break
    base = f"{author}{year}{word}"
    base = re.sub(r"[^a-z0-9]+", "", base.lower())
    return base or "ref"


def dedup_citekey(base: str, used: set) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 2
    while True:
        cand = f"{base}{i}"
        if cand not in used:
            used.add(cand)
            return cand
        i += 1


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


@dataclass
class EntryRecord:
    source_file: str
    source_id: str
    entry: Dict[str, str]


def read_bib(path: str) -> bibtexparser.bibdatabase.BibDatabase:
    with open(path, "r", encoding="utf-8") as f:
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        parser.ignore_nonstandard_types = False
        return parser.parse_file(f)


def write_bib(entries: List[Dict[str, str]], path: str) -> None:
    db = bibtexparser.bibdatabase.BibDatabase()
    db.entries = entries
    writer = BibTexWriter()
    writer.indent = "  "
    with open(path, "w", encoding="utf-8") as f:
        f.write(writer.write(db))


def collect_records(input_files: List[str]) -> List[EntryRecord]:
    records: List[EntryRecord] = []
    for p in input_files:
        db = read_bib(p)
        for e in db.entries:
            normalize_entry(e)
            sid = e.get("ID") or ""
            records.append(EntryRecord(source_file=p, source_id=sid, entry=e))
    return records


def build_clusters(records: List[EntryRecord]) -> List[List[int]]:
    n = len(records)
    uf = UnionFind(n)
    strong_key_to_idx: Dict[str, int] = {}
    fuzzy_key_of_idx: List[Optional[str]] = [None] * n
    has_strong: List[bool] = [False] * n

    for i, rec in enumerate(records):
        doi, arxiv, openreview = extract_ids_from_entry(rec.entry)
        strong_keys = []
        if doi:
            strong_keys.append(f"doi:{doi}")
        if arxiv:
            strong_keys.append(f"arxiv:{arxiv}")
        if openreview:
            strong_keys.append(f"openreview:{openreview}")

        for key in strong_keys:
            j = strong_key_to_idx.get(key)
            if j is None:
                strong_key_to_idx[key] = i
            else:
                uf.union(i, j)

        has_strong[i] = bool(strong_keys)
        fuzzy_key_of_idx[i] = fuzzy_signature(rec.entry)

    # Cluster no-strong entries by exact fuzzy signature.
    fuzzy_no_strong: Dict[str, int] = {}
    for i, fkey in enumerate(fuzzy_key_of_idx):
        if not fkey or has_strong[i]:
            continue
        j = fuzzy_no_strong.get(fkey)
        if j is None:
            fuzzy_no_strong[fkey] = i
        else:
            uf.union(i, j)

    # If two strong components share the same fuzzy signature, merge them.
    root_has_strong: Dict[int, bool] = {}
    for i in range(n):
        r = uf.find(i)
        root_has_strong[r] = root_has_strong.get(r, False) or has_strong[i]
    fuzzy_strong_root: Dict[str, int] = {}
    for i, fkey in enumerate(fuzzy_key_of_idx):
        if not fkey:
            continue
        r = uf.find(i)
        if not root_has_strong.get(r, False):
            continue
        prev = fuzzy_strong_root.get(fkey)
        if prev is None:
            fuzzy_strong_root[fkey] = r
        elif prev != r:
            uf.union(prev, r)

    # Attach no-strong components to strong components via fuzzy signature.
    root_has_strong = {}
    for i in range(n):
        r = uf.find(i)
        root_has_strong[r] = root_has_strong.get(r, False) or has_strong[i]
    fuzzy_strong_root = {}
    for i, fkey in enumerate(fuzzy_key_of_idx):
        if not fkey:
            continue
        r = uf.find(i)
        if root_has_strong.get(r, False):
            fuzzy_strong_root[fkey] = r
    for i, fkey in enumerate(fuzzy_key_of_idx):
        if not fkey:
            continue
        r = uf.find(i)
        if root_has_strong.get(r, False):
            continue
        target = fuzzy_strong_root.get(fkey)
        if target is not None:
            uf.union(i, target)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def choose_representative(indices: List[int], records: List[EntryRecord]) -> int:
    def rank(i: int) -> Tuple[int, int]:
        e = records[i].entry
        return (1 if has_venue_info(e) else 0, entry_completeness_score(e))

    return max(indices, key=rank)


def deduplicate_records(records: List[EntryRecord]) -> Tuple[List[Dict[str, str]], Dict[str, str], List[Dict[str, Any]]]:
    clusters = build_clusters(records)
    used_ids: set = set()
    merged_entries: List[Dict[str, str]] = []
    keymap: Dict[str, str] = {}
    report_rows: List[Dict[str, Any]] = []

    for cluster_idx, indices in enumerate(clusters, start=1):
        rep_i = choose_representative(indices, records)
        merged = deepcopy(records[rep_i].entry)
        for i in indices:
            if i == rep_i:
                continue
            merge_entries(merged, records[i].entry)
        normalize_entry(merged)

        base_id = merged.get("ID") or suggest_citekey(merged)
        canon_id = dedup_citekey(base_id, used_ids)
        merged["ID"] = canon_id
        merged_entries.append(merged)

        members = []
        for i in indices:
            rec = records[i]
            source_key = f"{rec.source_file}::{rec.source_id or '(no-id)'}@{i}"
            keymap[source_key] = canon_id
            doi, arxiv, openreview = extract_ids_from_entry(rec.entry)
            members.append(
                {
                    "file": rec.source_file,
                    "index": i,
                    "id": rec.source_id,
                    "doi": doi,
                    "arxiv": arxiv,
                    "openreview": openreview,
                }
            )

        report_rows.append(
            {
                "cluster": cluster_idx,
                "canonical_id": canon_id,
                "size": len(indices),
                "source": "dedup",
                "title": merged.get("title"),
                "members": members,
            }
        )

    merged_entries.sort(key=lambda e: e.get("ID", ""))
    return merged_entries, keymap, report_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="One or more collaborator .bib files")
    ap.add_argument("-o", "--output", default="dedup_merged.bib", help="Path of merged deduplicated .bib")
    ap.add_argument("--report", default="dedup_report.jsonl", help="JSONL cluster report path")
    ap.add_argument("--keymap", default="dedup_keymap.json", help="JSON mapping old key -> canonical key")
    ap.add_argument("--quiet", action="store_true", help="Reduce console logs")
    args = ap.parse_args()

    ensure_dependencies()

    records = collect_records(args.inputs)
    merged_entries, keymap, report_rows = deduplicate_records(records)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_bib(merged_entries, str(out_path))

    with open(args.report, "w", encoding="utf-8") as f:
        for row in report_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_files": args.inputs,
        "input_entries": len(records),
        "output_entries": len(merged_entries),
        "duplicates_removed": max(0, len(records) - len(merged_entries)),
        "map": keymap,
    }
    with open(args.keymap, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print(f"input entries:   {len(records)}")
        print(f"output entries:  {len(merged_entries)}")
        print(f"removed dups:    {max(0, len(records) - len(merged_entries))}")
        print(f"merged bib:      {out_path}")
        print(f"report jsonl:    {args.report}")
        print(f"key map json:    {args.keymap}")


if __name__ == "__main__":
    main()
