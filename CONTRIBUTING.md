# Contributing

Thanks for your interest in contributing to `pisinger-knapsack`.

## Development environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install                 # optional
```

## Before submitting changes

Run the same set of checks locally as CI does:

```bash
ruff check .
ruff format --check .
mypy
pytest
```

## Style and conventions

- Formatting and linting: **Ruff** (configuration in `pyproject.toml`).
- Typing: **mypy** in `strict` mode; the public API is fully typed.
- Tests: **pytest**; new code should be covered by tests.
- Commits: a short, imperative-mood summary.

## Releases

A release is created by pushing a `vX.Y.Z` tag following Semantic Versioning;
the `release` workflow builds the distributions and publishes them to PyPI.
Remember to update `CHANGELOG.md` and the `__version__` field.
