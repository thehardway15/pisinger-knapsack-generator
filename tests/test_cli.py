"""Tests for the command-line interface (R1-T7).

Executable specification for the generic CLI scaffold built on top of the
library. The CLI is the boundary where a ``seed`` is turned into a generator
(``numpy.random.default_rng(seed)``) and recorded as provenance -- the library
core stays seed-agnostic.

    pisinger-knapsack generate --n N --correlation TYPE --seed S \
        [--R R] [--offset O] [--meta KEY=VALUE ...] --out FILE
    pisinger-knapsack manifest build  --dir DIR --out FILE
    pisinger-knapsack manifest verify --dir DIR --manifest FILE

Contract:

* ``main(argv) -> int`` -- ``0`` on success, non-zero on a handled failure
  (e.g. a failed verification). ``argparse`` usage errors exit with code ``2``.
* ``generate`` writes one instance via ``save_instance`` and records the seed
  (plus any ``--meta`` pairs) in the file's ``metadata``. Same arguments produce
  a byte-identical file.
* ``manifest build`` indexes ``*.json`` files in ``--dir`` using the file stem as
  ``instance_id``. ``manifest verify`` returns ``0`` for an intact set and a
  non-zero code when a file no longer matches its checksum.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pisinger_knapsack import (
    CorrelationType,
    generate_instance,
    load_instance,
    read_manifest,
    read_metadata,
)
from pisinger_knapsack.cli import main

DEFAULT_R = 1000


def run_generate(
    out: Path,
    *,
    n: int = 20,
    correlation: str = "uncorrelated",
    seed: int = 123,
    extra: list[str] | None = None,
) -> int:
    """Invoke the ``generate`` subcommand and return its exit code."""
    argv = [
        "generate",
        "--n",
        str(n),
        "--correlation",
        correlation,
        "--seed",
        str(seed),
        "--out",
        str(out),
    ]
    if extra:
        argv += extra
    return main(argv)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def test_version_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def test_generate_writes_loadable_instance(tmp_path: Path) -> None:
    out = tmp_path / "instance.json"

    code = run_generate(out, n=20, correlation="uncorrelated", seed=123)

    assert code == 0
    assert out.exists()
    instance = load_instance(out)
    assert instance.n == 20
    assert instance.correlation_type is CorrelationType.UNCORRELATED


def test_generate_matches_library_for_same_seed(tmp_path: Path) -> None:
    out = tmp_path / "instance.json"

    run_generate(out, n=30, correlation="weakly", seed=777)

    expected = generate_instance(
        np.random.default_rng(777),
        n=30,
        correlation_type=CorrelationType.WEAKLY_CORRELATED,
        R=DEFAULT_R,
    )
    assert load_instance(out) == expected


def test_generate_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"

    run_generate(first, seed=5)
    run_generate(second, seed=5)

    assert first.read_bytes() == second.read_bytes()


def test_generate_records_seed_in_metadata(tmp_path: Path) -> None:
    out = tmp_path / "instance.json"

    run_generate(out, seed=4242)

    assert read_metadata(out)["seed"] == 4242


def test_generate_records_extra_meta(tmp_path: Path) -> None:
    out = tmp_path / "instance.json"

    run_generate(out, seed=1, extra=["--meta", "note=demo"])

    metadata = read_metadata(out)
    assert metadata["seed"] == 1
    assert metadata["note"] == "demo"


def test_generate_honours_offset_for_strongly(tmp_path: Path) -> None:
    out = tmp_path / "instance.json"

    run_generate(out, correlation="strongly", seed=9, extra=["--offset", "50"])

    instance = load_instance(out)
    assert all(v == w + 50 for v, w in zip(instance.values, instance.weights, strict=True))


def test_generate_rejects_unknown_correlation(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        run_generate(tmp_path / "x.json", correlation="bogus")

    assert exc.value.code == 2


def test_generate_requires_out(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--n", "20", "--correlation", "uncorrelated", "--seed", "1"])

    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# manifest build / verify
# ---------------------------------------------------------------------------


def _populate_dataset(tmp_path: Path) -> Path:
    """Generate a two-instance dataset directory and return it."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    run_generate(data_dir / "n20_uncorrelated.json", n=20, correlation="uncorrelated", seed=1)
    run_generate(data_dir / "n30_strongly.json", n=30, correlation="strongly", seed=2)
    return data_dir


def test_manifest_build_indexes_directory_by_stem(tmp_path: Path) -> None:
    data_dir = _populate_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.json"

    code = main(["manifest", "build", "--dir", str(data_dir), "--out", str(manifest_path)])

    assert code == 0
    manifest = read_manifest(manifest_path)
    ids = {entry["instance_id"] for entry in manifest["instances"]}
    assert ids == {"n20_uncorrelated", "n30_strongly"}


def test_manifest_verify_passes_for_intact_dataset(tmp_path: Path) -> None:
    data_dir = _populate_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    main(["manifest", "build", "--dir", str(data_dir), "--out", str(manifest_path)])

    code = main(["manifest", "verify", "--dir", str(data_dir), "--manifest", str(manifest_path)])

    assert code == 0


def test_manifest_verify_fails_for_tampered_dataset(tmp_path: Path) -> None:
    data_dir = _populate_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    main(["manifest", "build", "--dir", str(data_dir), "--out", str(manifest_path)])
    (data_dir / "n20_uncorrelated.json").write_text("{}\n", encoding="utf-8")

    code = main(["manifest", "verify", "--dir", str(data_dir), "--manifest", str(manifest_path)])

    assert code != 0
