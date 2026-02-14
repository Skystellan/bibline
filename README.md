# bibline

[中文文档](README.zh-CN.md)

`bibline` is a BibTeX toolkit for CS writing workflows:

- `rebib.py`: identifier-first cleaning/enrichment.
- `dedup_bib.py`: collaborator multi-file dedup and merge.

## Use `uv` For Dependency Management

### 1) Install and sync

```bash
uv sync
```

This creates a local virtual environment and installs dependencies from `pyproject.toml`.

### 2) Run cleaner

```bash
uv run python rebib.py references_a.bib references_b.bib -o cleaned
```

### 3) Run dedup across collaborators

```bash
uv run python dedup_bib.py alice.bib bob.bib charlie.bib \
  -o dedup_merged.bib \
  --report dedup_report.jsonl \
  --keymap dedup_keymap.json
```

## Optional: Set Zsh Shortcuts

If you use `zsh`, add shortcuts so you can run commands from anywhere.

### Option A: Two aliases

Add to `~/.zshrc`:

```zsh
alias rebiber='uv run --project /absolute/path/to/bibline python /absolute/path/to/bibline/rebib.py'
alias rebibdedup='uv run --project /absolute/path/to/bibline python /absolute/path/to/bibline/dedup_bib.py'
```

Example with this repo path:

```zsh
alias rebiber='uv run --project /Users/skystellan/ResearchTools/Rebiber python /Users/skystellan/ResearchTools/Rebiber/rebib.py'
alias rebibdedup='uv run --project /Users/skystellan/ResearchTools/Rebiber python /Users/skystellan/ResearchTools/Rebiber/dedup_bib.py'
```

### Option B: Single command (`rebiber`)

Add to `~/.zshrc`:

```zsh
rebiber() {
  local proj="/absolute/path/to/bibline"
  if [[ "$1" == "dedup" ]]; then
    shift
    uv run --project "$proj" python "$proj/dedup_bib.py" "$@"
  else
    uv run --project "$proj" python "$proj/rebib.py" "$@"
  fi
}
```

Then reload shell config:

```bash
source ~/.zshrc
```

Usage:

```bash
rebiber refs1.bib refs2.bib -o cleaned
rebiber dedup alice.bib bob.bib -o dedup_merged.bib
```

## What `rebib.py` Does

- Resolution priority: DOI -> arXiv DOI upgrade -> DBLP -> Crossref/OpenAlex/arXiv fallback.
- Keeps citekeys (`ID`) stable.
- Venue-aware: avoids degrading published entries into arXiv-only records.
- Conservative title-case protection for acronyms/mixed-case terms.
- Default URL retention for manual audit.
- Optional cross-file sync (`--no-sync` to disable).
- JSONL audit report.
- Progress + speed controls:
  - `--workers`
  - `--http-timeout`
  - `--http-retries`
  - `--http-backoff`
  - `--no-cache`
  - `--no-progress`

## What `dedup_bib.py` Does

- Strong-ID clustering: `doi`, `arXiv`, `OpenReview`.
- Fuzzy bridge for no-ID entries: `title + first author + year`.
- Representative selection favors venue entries, then completeness.
- Outputs:
  - merged deduplicated `.bib`
  - `dedup_report.jsonl`
  - `dedup_keymap.json` (old key token -> canonical key)

## Team Workflow

1. Each collaborator maintains their own `.bib`.
2. Run `rebib.py` to clean each source bib.
3. Run `dedup_bib.py` across all bib files.
4. Use `dedup_keymap.json` to update citekeys in paper sources if needed.

## CLI

### `rebib.py`

```bash
uv run python rebib.py [-h] [--inplace] [-o OUTDIR] [--report REPORT] [--no-sync]
                       [--drop-url-with-doi] [--workers WORKERS]
                       [--http-timeout HTTP_TIMEOUT] [--http-retries HTTP_RETRIES]
                       [--http-backoff HTTP_BACKOFF] [--no-cache] [--no-progress]
                       inputs [inputs ...]
```

### `dedup_bib.py`

```bash
uv run python dedup_bib.py [-h] [-o OUTPUT] [--report REPORT] [--keymap KEYMAP]
                           [--quiet]
                           inputs [inputs ...]
```

## Security Note

Do not commit API keys or tokens to this repository. Use environment variables.
