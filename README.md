# bibline

`bibline` is a practical BibTeX toolkit for CS paper writing:

- `rebib.py`: identifier-first BibTeX cleaning and metadata repair.
- `dedup_bib.py`: multi-file collaborator deduplication and merge.

It is designed for teams where each collaborator keeps their own `.bib` and you still want one clean, deduplicated final bibliography.

## Features

### `rebib.py` (clean + enrich)
- Identifier-first resolution pipeline: DOI -> arXiv DOI upgrade -> DBLP -> Crossref/OpenAlex/arXiv fallback.
- Keeps citekeys stable (`ID` is preserved).
- Venue-aware behavior: avoids degrading published entries into arXiv-only fields.
- Conservative title case protection for acronyms/mixed-case tokens.
- Optional URL retention policy (`--drop-url-with-doi`).
- Cross-file duplicate synchronization (default on).
- JSONL audit report for each entry.
- Progress display + ETA.
- Tunable speed controls (`--workers`, timeout/retries/backoff/cache).

### `dedup_bib.py` (merge + dedup across collaborators)
- Clusters duplicates by strong IDs (`doi`, `arXiv`, `OpenReview`).
- Bridges no-ID records with fuzzy signature (`title + first author + year`).
- Picks a representative record per cluster and merges missing fields.
- Prefers venue entries over preprint-style records.
- Writes:
  - merged deduplicated `.bib`
  - cluster report (`.jsonl`)
  - old-key -> canonical-key mapping (`.json`)

## Requirements

- Python 3.9+
- Packages:
  - `requests`
  - `bibtexparser`

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1) Clean one or more bib files

```bash
python rebib.py references_a.bib references_b.bib -o cleaned
```

Outputs:
- cleaned files under `cleaned/`
- default report: `rebib_report.jsonl`

### 2) Deduplicate across collaborators

```bash
python dedup_bib.py alice.bib bob.bib charlie.bib \
  -o dedup_merged.bib \
  --report dedup_report.jsonl \
  --keymap dedup_keymap.json
```

## CLI Reference

### `rebib.py`

```bash
python rebib.py [-h] [--inplace] [-o OUTDIR] [--report REPORT] [--no-sync]
                [--drop-url-with-doi] [--workers WORKERS]
                [--http-timeout HTTP_TIMEOUT] [--http-retries HTTP_RETRIES]
                [--http-backoff HTTP_BACKOFF] [--no-cache] [--no-progress]
                inputs [inputs ...]
```

Key flags:
- `--workers`: parallel processing workers for entries.
- `--http-timeout`, `--http-retries`, `--http-backoff`: network tuning.
- `--no-cache`: disable local HTTP cache.
- `--no-sync`: disable cross-file synchronization.
- `--drop-url-with-doi`: remove URL for DOI entries unless web/openreview.

### `dedup_bib.py`

```bash
python dedup_bib.py [-h] [-o OUTPUT] [--report REPORT] [--keymap KEYMAP]
                    [--quiet]
                    inputs [inputs ...]
```

Key flags:
- `-o, --output`: merged deduplicated output bib.
- `--report`: cluster report JSONL.
- `--keymap`: old-key -> canonical-key mapping JSON.

## Suggested Team Workflow

1. Each collaborator keeps their own local `.bib`.
2. Run `rebib.py` on each file to normalize and enrich metadata.
3. Run `dedup_bib.py` on all collaborator files to produce one final bibliography.
4. Use `dedup_keymap.json` to update citekeys in manuscript sources if needed.
5. Commit both final `.bib` and report artifacts for traceability.

## Output Artifacts

### `rebib_report.jsonl`
Per-entry audit trail with:
- source file
- citekey
- action (`updated`, `untouched`, `synced`, `error`)
- selected source path
- before/after summary
- debug hints

### `dedup_report.jsonl`
Per-cluster summary with:
- canonical citekey
- cluster size
- merged title
- all member records and extracted IDs

### `dedup_keymap.json`
- input/output counts
- duplicate removal count
- mapping from original key tokens (`file::id@index`) to canonical citekey

## Performance Notes

- Start with `--workers 2` to `--workers 6` depending on network quality.
- If you hit rate limits (429), reduce workers and retries.
- Keep cache enabled for iterative runs unless debugging resolution behavior.

## Troubleshooting

- `Missing dependencies: ...`:
  - install required packages with `pip install -r requirements.txt`.
- Many unresolved records:
  - verify internet access and API endpoints.
  - set `REBIB_EMAIL` for polite API usage.
- Published records still look preprint-like:
  - rerun after updating scripts; current logic is venue-aware and strips arXiv-only fields when venue exists.

## Repository Layout

```text
bibline/
  rebib.py
  dedup_bib.py
  README.md
  requirements.txt
  .gitignore
  CONTRIBUTING.md
  LICENSE
```

## Security

- Do not commit API keys, tokens, or personal secrets into this repository.
- Prefer environment variables for runtime configuration.

