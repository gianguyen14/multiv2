# Unified AIC Retrieval

Multimodal video retrieval system for AIC 2026.

## Current milestone

**M0 - Repository skeleton**

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md).

## Engineering rules

See [AGENTS.md](AGENTS.md).

## Development

Install the package in editable mode:

```bash
python -m pip install -e .
```

Run tests:

```bash
pytest
```

Verify imports:

```bash
python -c "import backend; import backend.app; print('imports: OK')"
```
