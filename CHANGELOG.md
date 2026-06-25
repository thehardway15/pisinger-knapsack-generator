# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-06-25

### Added
- 0/1 knapsack instance generator with three Pisinger correlation classes
  (uncorrelated, weakly, strongly), parameterized by range `R` and offset.
- Deterministic generation via an injected `numpy.random.Generator` with no
  global RNG state: the same seed always yields the same instance.
- `KnapsackInstance` data model and the `W = floor(0.5 * sum(weights))` capacity rule.
- Canonical JSON serialization (`save_instance` / `load_instance`) with sorted
  keys and an optional provenance `metadata` field.
- Dataset manifest with per-instance SHA-256 checksums and integrity
  verification (`build_manifest`, `write_manifest`, `read_manifest`,
  `verify_manifest`).
- CLI subcommands: `generate`, `manifest build`, `manifest verify`.
- Example dataset under `examples/` with a deterministic regeneration script.

### Changed
- Output is written as canonical UTF-8 bytes (no platform newline translation),
  so checksums are reproducible across operating systems.

### Removed
- Placeholder greeting function and the `--name` CLI flag from the initial skeleton.

## [0.1.0] - 2026-06-15

### Added
- Initial repository and project structure.
- Build configuration (Hatchling, PEP 621) and package metadata.
- Tooling: Ruff (lint + format), mypy (strict), pytest + coverage.
- CI workflow (Python 3.10–3.13 on Linux/macOS/Windows) and a PyPI release
  workflow via Trusted Publishing (OIDC).
