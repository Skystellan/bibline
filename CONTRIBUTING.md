# Contributing

## Scope

This project focuses on BibTeX cleaning and deduplication for CS paper writing workflows.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Basic Checks

Run syntax checks before committing:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile rebib.py dedup_bib.py
```

## Change Guidelines

- Keep scripts runnable as standalone CLI tools.
- Prefer deterministic and auditable behavior.
- Preserve citekey stability unless explicitly needed.
- Avoid destructive operations on user bibliography fields without clear policy.

## Commit Style

Use clear, task-based messages, for example:

- `feat: add collaborator bib dedup tool`
- `fix: avoid preprint downgrade for published venue entries`
- `docs: add README command reference`
