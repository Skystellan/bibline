#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebib2.py — Identifiers-first BibTeX fixer

Pipeline (default):
0) local normalize (title/author/pages), extract identifiers
1) resolve by DOI -> Crossref BibTeX (authoritative)
2) resolve by arXiv -> arXiv metadata; if DOI found, upgrade via Crossref
3) fallback search by title:
   - DBLP search (best for CS), fetch dblp .bib for the best hit
   - Crossref works?query.bibliographic=... to discover DOI
   - OpenAlex works?search=... to discover DOI (optional; supports API key)
   - arXiv title search as last resort
4) (optional, default on) sync duplicates across input files
5) output cleaned .bib + report.jsonl (auditable)

Notes:
- Keeps citekey (ID) unchanged.
- Keeps URL by default for human audit; optional `--drop-url-with-doi` restores minimal-field mode.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

try:
    import bibtexparser
    from bibtexparser.bwriter import BibTexWriter
except ImportError:
    bibtexparser = None
    BibTexWriter = None

# -----------------------------
# Config
# -----------------------------
DBLP_API_URL = "https://dblp.org/search/publ/api"  # format=json, q=..., h=...
CROSSREF_WORKS = "https://api.crossref.org/works"
ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_WORKS = "https://api.openalex.org/works"

DEFAULT_MAX_WORKERS = 6
DEFAULT_TIMEOUT = 12
DEFAULT_HTTP_RETRIES = 3
DEFAULT_HTTP_BACKOFF = 0.8
SIM_THRESHOLD_TITLE = 0.90  # tighten to reduce false positives
CACHE_DIR = Path(os.getenv("REBIB_CACHE_DIR", "~/.cache/rebib")).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Runtime-tunable HTTP behavior (set from CLI in main()).
HTTP_TIMEOUT = DEFAULT_TIMEOUT
HTTP_MAX_RETRIES = DEFAULT_HTTP_RETRIES
HTTP_BACKOFF = DEFAULT_HTTP_BACKOFF
HTTP_USE_CACHE = True

PRINT_LOCK = threading.Lock()

# Keep these fields only for web-ish sources; otherwise may be dropped/rewritten
WEB_HOST_HINTS = ("openreview.net", "github.com", "arxiv.org", "alphaxiv.org")

DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
ARXIV_NEW_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b", re.IGNORECASE)
ARXIV_OLD_RE = re.compile(r"\b([a-z\-]+(\.[A-Z]{2})?/\d{7})(v\d+)?\b", re.IGNORECASE)
OPENREVIEW_RE = re.compile(r"openreview\.net/(?:forum|pdf)\?id=([A-Za-z0-9_\-]+)")

# Some fields are noise in many exported bibs; but we DON'T delete before extracting ids.
JUNK_FIELDS_ALWAYS = {
    "bibsource", "biburl", "timestamp", "file",
    "keywords", "abstract", "owner", "review",
}

# For non-arXiv entries, we often want to drop these (but not if the final is arXiv).
ARXIV_ONLY_FIELDS = {"eprint", "eprinttype", "archiveprefix", "primaryclass"}

# Mixed-case or domain terms that should keep case under sentence-case .bst styles.
TITLE_FORCE_PROTECT = {
    "aaai",
    "acl",
    "arxiv",
    "bibtex",
    "colbert",
    "cvpr",
    "eccv",
    "emnlp",
    "fse",
    "github",
    "iclr",
    "icml",
    "icse",
    "ijcai",
    "kdd",
    "latex",
    "naacl",
    "neurips",
    "openai",
    "pytorch",
    "roberta",
    "sigir",
    "tex",
    "uist",
    "www",
}
TITLE_FORCE_UNPROTECT = set()

# -----------------------------
# Utilities: normalization
# -----------------------------
def norm_space(s: str) -> str:
    return " ".join((s or "").replace("\n", " ").split()).strip()

def strip_braces(s: str) -> str:
    return (s or "").replace("{", "").replace("}", "")

def normalize_title_for_match(title: str) -> str:
    t = strip_braces(norm_space(title)).lower()
    # remove punctuation-ish
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def normalize_pages(p: str) -> str:
    p = norm_space(p)
    # Idempotent: collapse any run of dash-like separators to a BibTeX page range '--'.
    p = re.sub(r"\s*[–—-]+\s*", "--", p)
    return p

def protect_acronyms_in_title(title: str) -> str:
    """
    Minimal bracing:
    - Protect all-caps acronyms (LLM, RAG, IR, SWE)
    - Protect letter/digit mixes (T5, GPT4)
    - Protect mixed-case tokens (OpenAI, TensorFlow, iPhone, RoBERTa), except plain TitleCase
    - Protect a force-list for domain-specific terms (e.g., LaTeX, NeurIPS)
    - For hyphenated tokens, only protect the parts that need protection.
    - Keep punctuation and spacing untouched by tokenizing words inside non-braced spans.
    - Keep already-braced spans unchanged.
    """
    if not title:
        return title

    word_re = re.compile(r"[A-Za-z0-9+#/]+(?:-[A-Za-z0-9+#/]+)*")

    def should_protect(seg: str) -> bool:
        seg = seg.strip()
        if not seg:
            return False
        low = seg.lower()
        if low in TITLE_FORCE_UNPROTECT:
            return False
        if low in TITLE_FORCE_PROTECT:
            return True
        # all-caps acronym-ish token, optionally ending with digits/symbols
        if re.fullmatch(r"[A-Z]{2,}[A-Z0-9+#/]*", seg):
            return True
        # letter/digit mix, e.g., T5 or GPT4
        if re.search(r"[A-Z].*\d|\d.*[A-Z]", seg):
            return True
        # mixed-case token (OpenAI, TensorFlow, iPhone, RoBERTa), but skip plain TitleCase
        if re.search(r"[A-Z]", seg) and re.search(r"[a-z]", seg):
            if not re.fullmatch(r"[A-Z][a-z]+", seg):
                return True
        # language/tool names that are commonly title-cased incorrectly
        if seg in {"C++", "C#", "F#"}:
            return True
        return False

    def protect_core(core: str) -> str:
        if "-" in core:
            parts = core.split("-")
            changed = False
            new_parts = []
            for part in parts:
                if should_protect(part):
                    new_parts.append("{" + part + "}")
                    changed = True
                else:
                    new_parts.append(part)
            return "-".join(new_parts) if changed else core
        if should_protect(core):
            return "{" + core + "}"
        return core

    def protect_non_braced_span(span: str) -> str:
        if not span:
            return span
        out = []
        last = 0
        for m in word_re.finditer(span):
            out.append(span[last:m.start()])
            out.append(protect_core(m.group(0)))
            last = m.end()
        out.append(span[last:])
        return "".join(out)

    # Split title into spans outside/inside braces, preserving all original characters.
    spans: List[Tuple[bool, str]] = []
    buf: List[str] = []
    depth = 0
    for ch in title:
        if ch == "{":
            if depth == 0 and buf:
                spans.append((False, "".join(buf)))
                buf = []
            depth += 1
            buf.append(ch)
            continue
        if ch == "}":
            buf.append(ch)
            if depth > 0:
                depth -= 1
                if depth == 0:
                    spans.append((True, "".join(buf)))
                    buf = []
            continue
        buf.append(ch)
    if buf:
        spans.append((depth > 0, "".join(buf)))

    out_chunks = []
    for is_braced, span in spans:
        if is_braced:
            out_chunks.append(span)
        else:
            out_chunks.append(protect_non_braced_span(span))
    return "".join(out_chunks)

def extract_year_int(year: Optional[str]) -> Optional[int]:
    if year is None:
        return None
    m = re.search(r"(?:19|20)\d{2}", str(year))
    return int(m.group(0)) if m else None

def clean_author_name_dblp(name: str) -> str:
    name = (name or "").strip()
    # DBLP disambiguation suffix: "Tong Zhang 0001"
    return re.sub(r"\s+\d{4}\s*$", "", name)

def author_lastnames(author_field: str, k: int = 2) -> List[str]:
    """
    Extract up to k last names from 'A and B and C' (BibTeX style).
    """
    if not author_field:
        return []
    parts = [p.strip() for p in author_field.split(" and ") if p.strip()]
    last = []
    for p in parts[:k]:
        # "Last, First" or "First Last"
        if "," in p:
            ln = p.split(",")[0].strip()
        else:
            ln = p.split()[-1].strip()
        if ln:
            last.append(ln.lower())
    return last

# -----------------------------
# HTTP with caching + backoff
# -----------------------------
def _cache_key(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> str:
    payload = {"url": url, "params": params or {}, "h": headers or {}}
    h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return h

def cache_get(key: str) -> Optional[Dict[str, Any]]:
    fp = CACHE_DIR / f"{key}.json"
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None

def cache_put(key: str, obj: Dict[str, Any]) -> None:
    fp = CACHE_DIR / f"{key}.json"
    try:
        fp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def http_get(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    backoff: Optional[float] = None,
    use_cache: Optional[bool] = None,
) -> Tuple[int, str, Dict[str, str]]:
    timeout = HTTP_TIMEOUT if timeout is None else timeout
    max_retries = HTTP_MAX_RETRIES if max_retries is None else max_retries
    backoff = HTTP_BACKOFF if backoff is None else backoff
    use_cache = HTTP_USE_CACHE if use_cache is None else use_cache

    key = _cache_key(url, params=params, headers=headers)
    if use_cache:
        cached = cache_get(key)
        if cached and "status" in cached and "text" in cached and "resp_headers" in cached:
            return int(cached["status"]), cached["text"], dict(cached["resp_headers"])

    last_exc = None
    for i in range(max_retries):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=timeout)
            text = resp.text
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (2 ** i))
                continue
            if use_cache:
                cache_put(key, {"status": resp.status_code, "text": text, "resp_headers": dict(resp.headers)})
            return resp.status_code, text, dict(resp.headers)
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (2 ** i))
    # final fallback: treat as error
    return 0, f"__ERROR__:{last_exc}", {}

# -----------------------------
# ID extraction
# -----------------------------
@dataclass
class Ids:
    doi: Optional[str] = None
    arxiv: Optional[str] = None
    openreview: Optional[str] = None

def sanitize_doi(doi: str) -> str:
    d = norm_space(doi)
    d = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", d, flags=re.IGNORECASE)
    prev = None
    while d and d != prev:
        prev = d
        d = d.strip()
        d = d.rstrip(".,;")
        d = d.strip("<>\"'")

    # Unwrap one or more outer bracket pairs when DOI is wrapped as a whole.
    for open_c, close_c in (("(", ")"), ("[", "]"), ("{", "}")):
        while d.startswith(open_c) and d.endswith(close_c):
            inner = d[1:-1].strip()
            if not inner:
                break
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
    """Return a normalized arXiv id WITHOUT version suffix (vN) if present."""
    arx = extract_arxiv_id(s)
    if not arx:
        return None
    return re.sub(r"v\d+$", "", arx.lower())

def extract_openreview_id(s: str) -> Optional[str]:
    if not s:
        return None
    m = OPENREVIEW_RE.search(s)
    return m.group(1) if m else None

def extract_ids_from_entry(entry: Dict[str, str]) -> Ids:
    # look across common fields
    blob_fields = ["doi", "url", "note", "howpublished", "eprint", "journal", "booktitle"]
    blob = " ".join([entry.get(f, "") for f in blob_fields if entry.get(f)])
    doi = extract_doi(blob)
    arxiv = arxiv_base_id(blob)
    openreview = extract_openreview_id(blob)
    return Ids(doi=doi, arxiv=arxiv, openreview=openreview)

# -----------------------------
# Resolvers
# -----------------------------
def parse_single_bibtex(bibtex_str: str) -> Optional[Dict[str, str]]:
    try:
        db = bibtexparser.loads(bibtex_str)
        if not db.entries:
            return None
        return db.entries[0]
    except Exception:
        return None

def crossref_bibtex_by_doi(session: requests.Session, doi: str, ua: str) -> Optional[Dict[str, str]]:
    # Crossref transform route; returns BibTeX string :contentReference[oaicite:5]{index=5}
    doi = sanitize_doi(doi)
    if not doi:
        return None
    url = f"{CROSSREF_WORKS}/{urllib.parse.quote(doi)}/transform/application/x-bibtex"
    headers = {"User-Agent": ua}
    status, text, _ = http_get(session, url, headers=headers)
    if status != 200 or text.startswith("__ERROR__"):
        return None
    return parse_single_bibtex(text)

def crossref_search_doi(session: requests.Session, title: str, first_author_ln: Optional[str], year: Optional[str], ua: str) -> Optional[str]:
    params = {"query.bibliographic": strip_braces(title), "rows": 5}
    y = extract_year_int(year)
    if y:
        params["filter"] = f"from-pub-date:{y-1}-01-01,until-pub-date:{y+1}-12-31"
    headers = {"User-Agent": ua}
    status, text, _ = http_get(session, CROSSREF_WORKS, params=params, headers=headers)
    if status != 200 or text.startswith("__ERROR__"):
        return None
    try:
        data = json.loads(text)
        items = data.get("message", {}).get("items", [])
        if not items:
            return None
        q = normalize_title_for_match(title)
        for it in items:
            it_title = (it.get("title") or [""])[0]
            r = difflib.SequenceMatcher(None, normalize_title_for_match(it_title), q).ratio()
            if r < SIM_THRESHOLD_TITLE:
                continue
            if first_author_ln:
                a = it.get("author") or []
                if a:
                    ln = (a[0].get("family") or "").lower()
                    if ln and ln != first_author_ln.lower():
                        # allow mismatch if ratio very high
                        if r < 0.96:
                            continue
            doi = sanitize_doi(it.get("DOI") or "")
            if doi:
                return doi
    except Exception:
        return None
    return None

def dblp_search_best_key(session: requests.Session, title: str, ua: str) -> Optional[str]:
    """
    Return best dblp 'key' from search hits (excluding CoRR).
    """
    headers = {"User-Agent": ua}
    target_title = normalize_title_for_match(title)
    query_candidates = [strip_braces(norm_space(title)), target_title]
    seen_queries = set()
    best_key = None
    best_r = 0.0

    for q in query_candidates:
        if not q or q in seen_queries:
            continue
        seen_queries.add(q)
        params = {"q": q, "format": "json", "h": 5}
        status, text, _ = http_get(session, DBLP_API_URL, params=params, headers=headers)
        if status != 200 or text.startswith("__ERROR__"):
            continue
        try:
            hits = json.loads(text).get("result", {}).get("hits", {}).get("hit", [])
        except Exception:
            continue
        for hit in hits:
            info = hit.get("info", {})
            # skip arXiv/CoRR
            if info.get("venue") == "CoRR" or "journals/corr" in (info.get("url") or ""):
                continue
            dblp_title = info.get("title") or ""
            r = difflib.SequenceMatcher(None, normalize_title_for_match(dblp_title), target_title).ratio()
            if r > best_r:
                best_r = r
                best_key = info.get("key")

    if best_key and best_r >= SIM_THRESHOLD_TITLE:
        return best_key
    return None

def dblp_bibtex_by_key(session: requests.Session, key: str, ua: str) -> Optional[Dict[str, str]]:
    # DBLP biburl convention: https://dblp.org/rec/<key>.bib :contentReference[oaicite:6]{index=6}
    url = f"https://dblp.org/rec/{key}.bib"
    headers = {"User-Agent": ua}
    status, text, _ = http_get(session, url, headers=headers)
    if status != 200 or text.startswith("__ERROR__"):
        return None
    ent = parse_single_bibtex(text)
    if ent and ent.get("author"):
        # de-disambiguate names
        authors = [clean_author_name_dblp(a.strip()) for a in ent["author"].split(" and ")]
        ent["author"] = " and ".join([a for a in authors if a])
    return ent

def arxiv_query(session: requests.Session, params: Dict[str, Any], ua: str) -> Optional[List[Dict[str, Any]]]:
    headers = {"User-Agent": ua}
    status, text, _ = http_get(session, ARXIV_API, params=params, headers=headers)
    if status != 200 or text.startswith("__ERROR__"):
        return None
    try:
        # parse Atom
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(text)
        entries = []
        for e in root.findall("atom:entry", ns):
            title = norm_space(e.findtext("atom:title", default="", namespaces=ns))
            aid = e.findtext("atom:id", default="", namespaces=ns)
            # id like http://arxiv.org/abs/XXXX.XXXXXvN
            arxiv_id = extract_arxiv_id(aid) or extract_arxiv_id(title)  # fallback
            authors = [norm_space(a.findtext("atom:name", default="", namespaces=ns)) for a in e.findall("atom:author", ns)]
            doi = e.findtext("arxiv:doi", default="", namespaces=ns).strip().lower() if e.find("arxiv:doi", ns) is not None else ""
            published = e.findtext("atom:published", default="", namespaces=ns)
            year = published[:4] if published and len(published) >= 4 else ""
            primary = e.findtext("arxiv:primary_category", default="", namespaces=ns)
            if not primary:
                pc = e.find("arxiv:primary_category", ns)
                if pc is not None:
                    primary = pc.attrib.get("term", "")
            entries.append({
                "title": title,
                "arxiv": arxiv_id,
                "authors": authors,
                "doi": doi or None,
                "year": year or None,
                "primary": primary or None,
                "id_url": aid,
            })
        return entries
    except Exception:
        return None

def arxiv_by_id(session: requests.Session, arxiv_id: str, ua: str) -> Optional[Dict[str, Any]]:
    params = {"id_list": arxiv_id, "start": 0, "max_results": 1}
    res = arxiv_query(session, params, ua=ua)
    if not res:
        return None
    return res[0]

def arxiv_search_by_title(session: requests.Session, title: str, ua: str) -> Optional[Dict[str, Any]]:
    q = strip_braces(norm_space(title))
    # arXiv title query: ti:"..."
    params = {"search_query": f'ti:"{q}"', "start": 0, "max_results": 5}
    cands = arxiv_query(session, params, ua=ua)
    if not cands:
        return None
    tgt = normalize_title_for_match(title)
    best = None
    best_r = 0.0
    for c in cands:
        r = difflib.SequenceMatcher(None, normalize_title_for_match(c.get("title", "")), tgt).ratio()
        if r > best_r:
            best_r, best = r, c
    if best and best_r >= SIM_THRESHOLD_TITLE:
        return best
    return None

def openalex_search_doi(session: requests.Session, title: str, ua: str) -> Optional[str]:
    """
    Optional: use OpenAlex to discover DOI from title.
    OpenAlex now recommends API key usage (set OPENALEX_API_KEY). :contentReference[oaicite:7]{index=7}
    """
    api_key = os.getenv("OPENALEX_API_KEY")
    params = {"search": strip_braces(title), "per-page": 5}
    if api_key:
        params["api_key"] = api_key
    # polite mailto if provided
    mailto = os.getenv("REBIB_EMAIL")
    if mailto:
        params["mailto"] = mailto
    headers = {"User-Agent": ua}
    status, text, _ = http_get(session, OPENALEX_WORKS, params=params, headers=headers)
    if status != 200 or text.startswith("__ERROR__"):
        return None
    try:
        data = json.loads(text)
        results = data.get("results", [])
        if not results:
            return None
        tgt = normalize_title_for_match(title)
        for it in results:
            it_title = it.get("title") or ""
            r = difflib.SequenceMatcher(None, normalize_title_for_match(it_title), tgt).ratio()
            if r < SIM_THRESHOLD_TITLE:
                continue
            doi = it.get("doi") or (it.get("ids") or {}).get("doi")
            if doi:
                doi = sanitize_doi(doi)
                return doi
    except Exception:
        return None
    return None

# -----------------------------
# Merge / finalize policy
# -----------------------------
def is_openreview_entry(entry: Dict[str, str]) -> bool:
    url = (entry.get("url") or "").lower()
    return "openreview.net" in url

def is_webish(entry: Dict[str, str]) -> bool:
    url = (entry.get("url") or "").lower()
    return any(h in url for h in WEB_HOST_HINTS)

def has_venue_info(entry: Dict[str, str]) -> bool:
    return bool(norm_space(entry.get("booktitle") or entry.get("journal") or ""))

def merge_canonical_into(entry: Dict[str, str], canon: Dict[str, str], keep_id: bool = True) -> None:
    """
    Merge canon fields into entry, keep citekey (ID) optionally.
    """
    eid = entry.get("ID")
    # copy entrytype from canon
    if canon.get("ENTRYTYPE"):
        entry["ENTRYTYPE"] = canon["ENTRYTYPE"]
    # overwrite key bibliographic fields if present
    for f in ["title", "author", "year", "booktitle", "journal", "volume", "number", "pages", "publisher", "doi"]:
        if canon.get(f):
            entry[f] = canon[f]

    # arXiv-only fields: merge if present in canon, otherwise clear stale values
    for f in ["eprinttype", "eprint", "archiveprefix", "primaryclass"]:
        if canon.get(f):
            entry[f] = canon[f]
        else:
            entry.pop(f, None)
    # Keep URL for human audit; prefer canonical URL if present.
    if canon.get("url"):
        entry["url"] = canon["url"]
    if keep_id and eid:
        entry["ID"] = eid

def apply_local_normalization(entry: Dict[str, str]) -> None:
    # normalize pages and title brace protection
    if entry.get("pages"):
        entry["pages"] = normalize_pages(entry["pages"])
    if entry.get("title"):
        entry["title"] = protect_acronyms_in_title(norm_space(entry["title"]))
    if entry.get("doi"):
        doi = sanitize_doi(entry["doi"])
        if doi:
            entry["doi"] = doi
        else:
            entry.pop("doi", None)

    # normalize arXiv eprint (drop version vN for stable de-dup)
    if entry.get("eprint"):
        base = arxiv_base_id(entry["eprint"])
        if base:
            entry["eprint"] = base
    if entry.get("author"):
        entry["author"] = norm_space(entry["author"])
    if entry.get("year"):
        entry["year"] = str(entry["year"]).strip()

def cleanup_fields(entry: Dict[str, str], *, drop_url_with_doi: bool = False) -> None:
    # Always drop obvious junk
    for k in list(entry.keys()):
        if k.lower() in JUNK_FIELDS_ALWAYS:
            entry.pop(k, None)

    has_venue = has_venue_info(entry)
    is_arxiv_preprint = (
        (entry.get("eprinttype", "").lower() == "arxiv" or entry.get("archiveprefix", "").lower() == "arxiv")
        and (not has_venue)
    )

    # Drop arXiv-only fields unless this is actually a preprint-only record.
    if not is_arxiv_preprint:
        for k in list(entry.keys()):
            if k.lower() in ARXIV_ONLY_FIELDS:
                entry.pop(k, None)

    # Optional minimal-field policy.
    if drop_url_with_doi and entry.get("doi") and (not is_webish(entry)) and (not is_openreview_entry(entry)):
        entry.pop("url", None)

def build_arxiv_bib(entry: Dict[str, str], meta: Dict[str, Any]) -> Dict[str, str]:
    """
    Construct a reasonable arXiv preprint record when no DOI is available.
    """
    out: Dict[str, str] = {}
    out["ENTRYTYPE"] = entry.get("ENTRYTYPE") or "misc"
    out["title"] = protect_acronyms_in_title(meta.get("title") or entry.get("title") or "")
    if meta.get("authors"):
        out["author"] = " and ".join(meta["authors"])
    if meta.get("year"):
        out["year"] = str(meta["year"])
    arx_raw = meta.get("arxiv") or extract_arxiv_id(entry.get("url", "")) or extract_arxiv_id(entry.get("note", ""))
    arx = arxiv_base_id(arx_raw or "")
    if arx:
        out["eprinttype"] = "arxiv"
        out["eprint"] = arx
        out["archiveprefix"] = "arXiv"
        if meta.get("primary"):
            out["primaryclass"] = meta["primary"]
        # Prefer preserving an existing arxiv.org URL (often includes version vN)
        orig_url = entry.get("url", "")
        if orig_url and "arxiv.org" in orig_url.lower() and extract_arxiv_id(orig_url):
            out["url"] = orig_url
        else:
            out["url"] = meta.get("id_url") or f"https://arxiv.org/abs/{(arx_raw or arx)}"
    return out

# -----------------------------
# Core: resolve one entry
# -----------------------------
def resolve_entry(session: requests.Session, entry: Dict[str, str], ua: str) -> Tuple[Optional[Dict[str, str]], str, Dict[str, Any]]:
    """
    Returns: (canonical_entry_or_none, source_tag, debug_info)
    source_tag: "crossref:doi" | "arxiv:doi-upgrade" | "dblp" | "crossref:title" | "openalex:title" | "arxiv:title" | "none"
    """
    ids = extract_ids_from_entry(entry)
    dbg: Dict[str, Any] = {"ids": ids.__dict__}

    title = entry.get("title") or ""
    year = entry.get("year")
    has_venue = has_venue_info(entry)
    first_ln = author_lastnames(entry.get("author", ""), k=1)
    first_ln = first_ln[0] if first_ln else None
    arxiv_meta_for_fallback: Optional[Dict[str, Any]] = None

    # 1) DOI -> Crossref BibTeX
    if ids.doi:
        canon = crossref_bibtex_by_doi(session, ids.doi, ua=ua)
        if canon:
            dbg["doi"] = ids.doi
            return canon, "crossref:doi", dbg

    # 2) arXiv id -> if DOI exists, upgrade via Crossref.
    # If no DOI, defer preprint fallback until published-source lookups fail.
    if ids.arxiv:
        meta = arxiv_by_id(session, ids.arxiv, ua=ua)
        if meta:
            dbg["arxiv_meta"] = meta
            if meta.get("doi"):
                canon = crossref_bibtex_by_doi(session, meta["doi"], ua=ua)
                if canon:
                    return canon, "arxiv:doi-upgrade", dbg
            arxiv_meta_for_fallback = meta

    # 3) Title fallback: DBLP
    if title:
        key = dblp_search_best_key(session, title, ua=ua)
        dbg["dblp_key"] = key
        if key:
            dblp_ent = dblp_bibtex_by_key(session, key, ua=ua)
            if dblp_ent:
                # If dblp has DOI, prefer Crossref skeleton
                doi = extract_doi(dblp_ent.get("doi", "") or dblp_ent.get("url", "") or "")
                if doi:
                    canon = crossref_bibtex_by_doi(session, doi, ua=ua)
                    if canon:
                        return canon, "crossref:doi(from-dblp)", dbg
                return dblp_ent, "dblp", dbg

    # 4) Crossref title search -> DOI -> Crossref BibTeX
    if title:
        doi = crossref_search_doi(session, title, first_author_ln=first_ln, year=year, ua=ua)
        dbg["crossref_doi_from_title"] = doi
        if doi:
            canon = crossref_bibtex_by_doi(session, doi, ua=ua)
            if canon:
                return canon, "crossref:title", dbg

    # 5) OpenAlex title search -> DOI -> Crossref BibTeX (optional)
    if title:
        doi = openalex_search_doi(session, title, ua=ua)
        dbg["openalex_doi_from_title"] = doi
        if doi:
            canon = crossref_bibtex_by_doi(session, doi, ua=ua)
            if canon:
                return canon, "openalex:title->crossref", dbg

    # 6) arXiv title search
    if title:
        meta = arxiv_search_by_title(session, title, ua=ua)
        dbg["arxiv_title_meta"] = meta
        if meta:
            if meta.get("doi"):
                canon = crossref_bibtex_by_doi(session, meta["doi"], ua=ua)
                if canon:
                    return canon, "arxiv:title->crossref", dbg
            if not has_venue:
                return build_arxiv_bib(entry, meta), "arxiv:title-preprint", dbg
            dbg["arxiv_preprint_skipped"] = "has-venue"

    # 7) final arXiv fallback by id (only if still no published venue info)
    if arxiv_meta_for_fallback and (not has_venue):
        return build_arxiv_bib(entry, arxiv_meta_for_fallback), "arxiv:preprint", dbg
    if arxiv_meta_for_fallback and has_venue:
        dbg["arxiv_preprint_skipped"] = "has-venue"

    return None, "none", dbg

# -----------------------------
# Duplicate sync across files
# -----------------------------
def work_fingerprint(entry: Dict[str, str]) -> str:
    doi = (entry.get("doi") or "").lower().strip()
    if doi:
        return f"doi:{doi}"
    # arXiv
    eprint = (entry.get("eprint") or "")
    arx = arxiv_base_id(eprint) or arxiv_base_id(entry.get("url", "") or "")
    if arx:
        return f"arxiv:{arx}"
    # openreview id
    oid = extract_openreview_id(entry.get("url", "") or "")
    if oid:
        return f"openreview:{oid}"
    # fallback: title + first author + year
    t = normalize_title_for_match(entry.get("title", ""))
    ln = author_lastnames(entry.get("author", ""), k=1)
    ln = ln[0] if ln else ""
    y = (entry.get("year") or "").strip()
    return f"fuzzy:{t}|{ln}|{y}"

def entry_completeness_score(entry: Dict[str, str]) -> int:
    score = 0
    if entry.get("doi"): score += 100
    if entry.get("pages"): score += 20
    if entry.get("booktitle") or entry.get("journal"): score += 20
    if entry.get("author"): score += 10
    if entry.get("year"): score += 5
    score += len(entry.keys())
    return score

def sync_cluster(entries: List[Dict[str, str]]) -> None:
    # Prefer published venue entries over preprints, then pick most complete.
    rep = max(entries, key=lambda e: (1 if has_venue_info(e) else 0, entry_completeness_score(e)))
    rep_core = {k: rep.get(k) for k in ["ENTRYTYPE","title","author","year","booktitle","journal","volume","number","pages","publisher","doi","eprinttype","eprint","archiveprefix","primaryclass","url"] if rep.get(k)}
    for e in entries:
        eid = e.get("ID")
        canon = dict(rep_core)
        if has_venue_info(e):
            for f in ["eprinttype", "eprint", "archiveprefix", "primaryclass"]:
                canon.pop(f, None)
        merge_canonical_into(e, canon, keep_id=True)
        if eid:
            e["ID"] = eid

# -----------------------------
# Process bib files
# -----------------------------
def build_user_agent() -> str:
    email = os.getenv("REBIB_EMAIL") or "unknown@example.com"
    return f"rebib2/1.0 (mailto:{email})"

def format_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(seconds + 0.5)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

class ProgressPrinter:
    def __init__(self, total: int, *, enabled: bool, min_interval: float = 0.2) -> None:
        self.total = max(0, int(total))
        self.enabled = enabled and self.total > 0
        self.min_interval = min_interval
        self.started_at = time.time()
        self.last_print = 0.0
        self.last_len = 0

    def update(self, done: int, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and (now - self.last_print) < self.min_interval and done < self.total:
            return
        elapsed = max(1e-6, now - self.started_at)
        done = min(max(done, 0), self.total)
        rate = done / elapsed if done else 0.0
        remaining = self.total - done
        eta = (remaining / rate) if rate > 1e-9 else None
        pct = (done / self.total) * 100.0
        line = f"processing {done}/{self.total} ({pct:5.1f}%) | {rate:4.2f} it/s | ETA {format_eta(eta)}"
        pad = max(0, self.last_len - len(line))
        with PRINT_LOCK:
            print("\r" + line + (" " * pad), end="", flush=True, file=sys.stderr)
        self.last_len = len(line)
        self.last_print = now

    def finish(self) -> None:
        if not self.enabled:
            return
        self.update(self.total, force=True)
        with PRINT_LOCK:
            print("", file=sys.stderr, flush=True)

def report_core(entry: Dict[str, str]) -> Dict[str, Any]:
    return {
        "doi": entry.get("doi"),
        "year": entry.get("year"),
        "venue": entry.get("booktitle") or entry.get("journal"),
        "url": entry.get("url"),
        "eprint": entry.get("eprint"),
        "pages": entry.get("pages"),
    }

def ensure_dependencies() -> None:
    missing = []
    if requests is None:
        missing.append("requests")
    if bibtexparser is None or BibTexWriter is None:
        missing.append("bibtexparser")
    if missing:
        raise SystemExit(
            "Missing dependencies: "
            + ", ".join(missing)
            + ". Install them first, e.g. `pip install requests bibtexparser`."
        )

def process_one_entry(
    session: requests.Session,
    src: str,
    e: Dict[str, str],
    ua: str,
    *,
    drop_url_with_doi: bool = False,
) -> Dict[str, Any]:
    before = deepcopy(e)
    try:
        apply_local_normalization(e)
        # extract ids before any deletion
        canon, source_tag, dbg = resolve_entry(session, e, ua=ua)

        changed = False
        if canon:
            # don't import canon's ID
            canon = deepcopy(canon)
            canon.pop("ID", None)
            merge_canonical_into(e, canon, keep_id=True)
            changed = True

        # keep OpenReview url always if present
        if is_openreview_entry(before) and before.get("url"):
            e["url"] = before["url"]

        apply_local_normalization(e)
        cleanup_fields(e, drop_url_with_doi=drop_url_with_doi)

        # ensure year is string
        if e.get("year"):
            e["year"] = str(e["year"]).strip()

        return {
            "file": src,
            "id": e.get("ID"),
            "action": "updated" if changed else "untouched",
            "source": source_tag,
            "before": report_core(before),
            "after": report_core(e),
            "debug": dbg,
        }
    except Exception as exc:
        return {
            "file": src,
            "id": e.get("ID"),
            "action": "error",
            "source": "exception",
            "before": report_core(before),
            "after": report_core(e),
            "debug": {"error": repr(exc)},
        }

def process_entries(
    session: requests.Session,
    all_entries: List[Tuple[str, Dict[str, str]]],
    ua: str,
    *,
    drop_url_with_doi: bool = False,
    workers: int = 1,
    show_progress: bool = True,
) -> Tuple[List[Tuple[str, Dict[str, str]]], List[Dict[str, Any]]]:
    """
    all_entries: list of (source_file, entry_dict)
    returns: updated entries + report rows
    """
    total = len(all_entries)
    report_rows: List[Optional[Dict[str, Any]]] = [None] * total
    workers = max(1, int(workers))
    progress = ProgressPrinter(total, enabled=show_progress)

    if workers == 1:
        for idx, (src, e) in enumerate(all_entries):
            report_rows[idx] = process_one_entry(
                session,
                src,
                e,
                ua,
                drop_url_with_doi=drop_url_with_doi,
            )
            progress.update(idx + 1)
        progress.finish()
        return all_entries, [r for r in report_rows if r is not None]

    thread_local = threading.local()

    def worker(i: int, src: str, entry: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
        t_session = getattr(thread_local, "session", None)
        if t_session is None:
            t_session = requests.Session()
            t_session.headers.update({"User-Agent": ua})
            thread_local.session = t_session
        row = process_one_entry(
            t_session,
            src,
            entry,
            ua,
            drop_url_with_doi=drop_url_with_doi,
        )
        return i, row

    future_to_idx = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (src, e) in enumerate(all_entries):
            fut = ex.submit(worker, i, src, e)
            future_to_idx[fut] = i

        done = 0
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                idx, row = fut.result()
            except Exception as exc:
                src, e = all_entries[i]
                idx = i
                row = {
                    "file": src,
                    "id": e.get("ID"),
                    "action": "error",
                    "source": "exception",
                    "before": report_core(e),
                    "after": report_core(e),
                    "debug": {"error": repr(exc)},
                }
            report_rows[idx] = row
            done += 1
            progress.update(done)

    progress.finish()
    return all_entries, [r for r in report_rows if r is not None]

def read_bib(path: str) -> bibtexparser.bibdatabase.BibDatabase:
    ensure_dependencies()
    with open(path, "r", encoding="utf-8") as f:
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        parser.ignore_nonstandard_types = False
        return parser.parse_file(f)

def write_bib(db: bibtexparser.bibdatabase.BibDatabase, path: str) -> None:
    ensure_dependencies()
    writer = BibTexWriter()
    writer.indent = "  "
    with open(path, "w", encoding="utf-8") as f:
        f.write(writer.write(db))

def main() -> None:
    global HTTP_TIMEOUT, HTTP_MAX_RETRIES, HTTP_BACKOFF, HTTP_USE_CACHE

    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="One or more .bib files")
    ap.add_argument("--inplace", action="store_true", help="Overwrite inputs")
    ap.add_argument("-o", "--outdir", default=None, help="Output directory (if not inplace)")
    ap.add_argument("--report", default="rebib_report.jsonl", help="Write JSONL report here")
    ap.add_argument("--no-sync", action="store_true", help="Disable cross-file duplicate sync")
    ap.add_argument("--drop-url-with-doi", action="store_true", help="Drop URL for DOI entries unless web/openreview")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers for entry processing")
    ap.add_argument("--http-timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds")
    ap.add_argument("--http-retries", type=int, default=DEFAULT_HTTP_RETRIES, help="HTTP retries on transient errors")
    ap.add_argument("--http-backoff", type=float, default=DEFAULT_HTTP_BACKOFF, help="HTTP retry backoff base seconds")
    ap.add_argument("--no-cache", action="store_true", help="Disable on-disk HTTP cache")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress display")
    args = ap.parse_args()

    ensure_dependencies()

    workers = max(1, args.workers)
    HTTP_TIMEOUT = max(1, args.http_timeout)
    HTTP_MAX_RETRIES = max(0, args.http_retries)
    HTTP_BACKOFF = max(0.0, args.http_backoff)
    HTTP_USE_CACHE = not args.no_cache

    ua = build_user_agent()
    session = requests.Session()
    session.headers.update({"User-Agent": ua})

    # load all bibs
    dbs: Dict[str, bibtexparser.bibdatabase.BibDatabase] = {}
    all_entries: List[Tuple[str, Dict[str, str]]] = []
    for p in args.inputs:
        db = read_bib(p)
        dbs[p] = db
        for e in db.entries:
            all_entries.append((p, e))

    # process
    all_entries, report_rows = process_entries(
        session,
        all_entries,
        ua=ua,
        drop_url_with_doi=args.drop_url_with_doi,
        workers=workers,
        show_progress=(not args.no_progress) and sys.stderr.isatty(),
    )

    src_by_entry_id = {id(e): src for src, e in all_entries}

    # sync duplicates across files (default on)
    if not args.no_sync:
        buckets: Dict[str, List[Dict[str, str]]] = {}
        for _, e in all_entries:
            fp = work_fingerprint(e)
            buckets.setdefault(fp, []).append(e)
        for _, group in buckets.items():
            if len(group) >= 2:
                before_group = [deepcopy(e) for e in group]
                sync_cluster(group)
                for before, after in zip(before_group, group):
                    if before == after:
                        continue
                    report_rows.append({
                        "file": src_by_entry_id.get(id(after), ""),
                        "id": after.get("ID"),
                        "action": "synced",
                        "source": "sync:cluster",
                        "before": report_core(before),
                        "after": report_core(after),
                        "debug": {"fingerprint": work_fingerprint(after)},
                    })

    # write outputs
    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    for src, db in dbs.items():
        if args.inplace:
            out_path = src
        else:
            if outdir:
                out_path = str(outdir / Path(src).name)
            else:
                out_path = f"cleaned_{Path(src).name}"
        write_bib(db, out_path)
        with PRINT_LOCK:
            print(f"✅ wrote: {out_path}")

    # write report
    with open(args.report, "w", encoding="utf-8") as f:
        for row in report_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"🧾 report: {args.report}")
    print(f"🗂 cache:  {CACHE_DIR}")

if __name__ == "__main__":
    main()
