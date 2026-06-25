# pisinger-knapsack

[![CI](https://github.com/thehardway15/pisinger-knapsack-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/thehardway15/pisinger-knapsack-generator/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pisinger-knapsack.svg)](https://pypi.org/project/pisinger-knapsack/)
[![Python](https://img.shields.io/pypi/pyversions/pisinger-knapsack.svg)](https://pypi.org/project/pisinger-knapsack/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A deterministic generator of **0/1 knapsack problem instances** based on
Pisinger's scheme — three value–weight correlation classes, an explicit seed,
canonical JSON output and a dataset manifest with checksums. Built so that an
experiment's input data is **fully reproducible** and verifiable.

## Why

Benchmark results are only trustworthy if the inputs can be regenerated exactly.
This generator is designed around that requirement:

- **Deterministic.** Generation is driven by an injected `numpy.random.Generator`
  and never touches global RNG state — the same seed always yields the same
  instance, regardless of call order.
- **Canonical & portable.** Instances serialize to byte-stable JSON (sorted
  keys, no platform newline translation), so a file's SHA-256 checksum is
  identical on every operating system.
- **Verifiable.** A manifest indexes a dataset and lets you confirm later that
  no instance file has changed.

## Installation

```bash
pip install pisinger-knapsack
```

Development version (from a checkout):

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+ and NumPy 2.x.

## Quick start (Python)

```python
import numpy as np
from pisinger_knapsack import (
    CorrelationType,
    generate_instance,
    save_instance,
    load_instance,
)

# Inject an explicit generator — this is what makes results reproducible.
rng = np.random.default_rng(20260101)

instance = generate_instance(rng, n=20, correlation_type=CorrelationType.WEAKLY_CORRELATED)
print(instance.n, instance.R, instance.capacity)   # 20 1000 <W>
print(instance.values, instance.weights)           # aligned int64 arrays

save_instance(instance, "instance.json", metadata={"seed": 20260101})
assert load_instance("instance.json") == instance
```

`generate_instance(rng, n, correlation_type, R=1000, offset=None)` returns a
frozen `KnapsackInstance(n, R, correlation_type, values, weights, capacity)`.
The capacity follows the **"50% knapsack" rule**, `W = floor(0.5 · Σ weights)`.

## Correlation classes

Given a weight `w ~ U(1, R)` and an offset `d` (default `R // 10`):

| Class               | `correlation_type` | Value `v`                                  |
| ------------------- | ------------------ | ------------------------------------------ |
| Uncorrelated        | `uncorrelated`     | `v ~ U(1, R)`, independent of `w`          |
| Weakly correlated   | `weakly`           | `v ~ U(w − d, w + d)`, clipped to `v ≥ 1`  |
| Strongly correlated | `strongly`         | `v = w + d`                                |

Weights always lie in `[1, R]`; values are always `≥ 1` and may exceed `R` for
the weakly and strongly correlated classes (by design). Correlation strength
increases across the classes, which is what drives the difficulty spread.

## Command-line interface

### `generate`

```bash
pisinger-knapsack generate --n 20 --correlation weakly --seed 1 --out instance.json
```

| Flag                | Required | Default    | Meaning                                       |
| ------------------- | -------- | ---------- | --------------------------------------------- |
| `--n`               | yes      | —          | Number of items                               |
| `--correlation`     | yes      | —          | `uncorrelated` \| `weakly` \| `strongly`      |
| `--seed`            | yes      | —          | RNG seed (recorded in the file's `metadata`)  |
| `--R`               | no       | `1000`     | Data-range coefficient                        |
| `--offset`          | no       | `R // 10`  | Value–weight offset                           |
| `--out`             | yes      | —          | Output file path                              |
| `--meta KEY=VALUE`  | no       | —          | Extra provenance, repeatable                  |

### `manifest`

```bash
pisinger-knapsack manifest build  --dir ./data --out ./data/manifest.json
pisinger-knapsack manifest verify --dir ./data --manifest ./data/manifest.json
```

`manifest verify` exits with a non-zero status if any instance file no longer
matches the checksum recorded in the manifest.

## Output format

Each instance is a JSON object with sorted keys (arrays shown compact here; on
disk each element is on its own line):

```json
{
    "R": 1000,
    "capacity": 812,
    "correlation_type": "weakly",
    "metadata": { "seed": 1 },
    "n": 3,
    "schema_version": 1,
    "values": [415, 720, 88],
    "weights": [402, 731, 95]
}
```

`metadata` is optional, free-form provenance — the library copies it verbatim
and assigns no meaning to its contents. The manifest indexes a directory of
instances; the `instance_id` is the file stem, so the caller controls naming:

```json
{
    "schema_version": 1,
    "instances": [
        {
            "R": 1000,
            "capacity": 812,
            "checksum": "sha256:…",
            "correlation_type": "weakly",
            "instance_id": "n20_weakly",
            "metadata": { "seed": 1 },
            "n": 20
        }
    ]
}
```

## Examples

The [`examples/data`](examples/data) directory ships a small, deterministic
dataset (two sizes × three correlations) and its manifest. Regenerate it
byte-for-byte at any time:

```bash
bash examples/generate.sh
```

A test ([`tests/test_examples.py`](tests/test_examples.py)) regenerates these
files and fails if they drift from what the current generator produces.

## Public API

Re-exported from the top-level package:

- `CorrelationType`, `KnapsackInstance`
- `generate_instance`, `capacity`
- `to_dict`, `from_dict`, `save_instance`, `load_instance`, `read_metadata`
- `build_manifest`, `write_manifest`, `read_manifest`, `verify_manifest`,
  `file_checksum`, `instance_checksum`

## Development

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

ruff check .          # lint
ruff format .         # formatting
mypy                  # type checking (strict)
pytest                # tests
```

Optional pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## References

The correlation classes and the instance design follow David Pisinger's work on
hard knapsack instances:

- D. Pisinger. *Where are the hard knapsack problems?* Computers & Operations
  Research, 32(9):2271–2284, 2005.
  doi:[10.1016/j.cor.2004.03.002](https://doi.org/10.1016/j.cor.2004.03.002)
- D. Pisinger. *Optimization codes* (knapsack instance generators).
  <http://hjemmesider.diku.dk/~pisinger/codes.html>

## License

[MIT](LICENSE) © Damian Wiśniewski
