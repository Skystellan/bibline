# Contributing

## Scope

This project focuses on BibTeX cleaning and deduplication for CS paper writing workflows.

## Development Setup

```bash
uv sync
```

## Basic Checks

Run syntax checks before committing:

```bash
uv run python -m py_compile rebib.py dedup_bib.py
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
