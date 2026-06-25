"""Command-line interface for the package."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pisinger_knapsack import __version__

from .constants import DEFAULT_R
from .correlation import CorrelationType
from .generator import generate_instance
from .manifest import file_checksum, read_manifest, verify_manifest, write_manifest
from .serialization import save_instance


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``pisinger-knapsack`` argument parser with its subcommands."""
    parser = argparse.ArgumentParser(
        prog="pisinger-knapsack",
        description="0/1 knapsack instance generator based on Pisinger's scheme.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subcommands = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen = subcommands.add_parser("generate", help="Generate a single instance.")
    gen.add_argument("--n", type=int, required=True)
    gen.add_argument("--correlation", required=True, choices=[c.value for c in CorrelationType])
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--R", type=int, default=DEFAULT_R)
    gen.add_argument("--offset", type=int, default=None)
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE")
    gen.set_defaults(func=_cmd_generate)

    # --- manifest (z własnymi podkomendami) ---
    man = subcommands.add_parser("manifest", help="Build or verify a manifest.")
    man_sub = man.add_subparsers(dest="manifest_command", required=True)

    build = man_sub.add_parser("build")
    build.add_argument("--dir", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.set_defaults(func=_cmd_manifest_build)

    verify = man_sub.add_parser("verify")
    verify.add_argument("--dir", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.set_defaults(func=_cmd_manifest_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Argument list; defaults to ``sys.argv``.

    Returns:
        Process exit code (``0`` on success, non-zero on a handled failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def _parse_meta(pairs: list[str]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` items from ``--meta`` into a mapping.

    Raises:
        SystemExit: If an item is not of the form ``KEY=VALUE``.
    """
    meta: dict[str, str] = {}
    for item in pairs:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--meta expects KEY=VALUE, got: {item!r}")
        meta[key] = value
    return meta


def _instance_files(directory: Path, *, exclude: Path | None = None) -> dict[str, Path]:
    """Map ``instance_id -> path`` for every ``*.json`` file in ``directory``.

    The file stem is used as the instance id. ``exclude`` (e.g. the manifest
    output path) is skipped if it happens to live inside ``directory``.
    """
    excluded = exclude.resolve() if exclude is not None else None
    return {
        path.stem: path
        for path in sorted(directory.glob("*.json"))
        if excluded is None or path.resolve() != excluded
    }


def _cmd_generate(args: argparse.Namespace) -> int:
    """Generate a single instance and write it to ``--out``."""
    rng = np.random.default_rng(args.seed)
    instance = generate_instance(
        rng,
        n=args.n,
        correlation_type=CorrelationType(args.correlation),
        R=args.R,
        offset=args.offset,
    )
    metadata = {"seed": args.seed, **_parse_meta(args.meta)}
    out = Path(args.out)
    save_instance(instance, out, metadata=metadata)

    print(
        f"Wrote {args.correlation} instance "
        f"(n={instance.n}, R={instance.R}, capacity={instance.capacity}) "
        f"to {out} [{file_checksum(out)}]"
    )
    return 0


def _cmd_manifest_build(args: argparse.Namespace) -> int:
    """Build a manifest indexing every instance file in ``--dir``."""
    out = Path(args.out)
    files = _instance_files(Path(args.dir), exclude=out)
    write_manifest(files, out)

    print(f"Wrote manifest for {len(files)} instance(s) to {out}")
    return 0


def _cmd_manifest_verify(args: argparse.Namespace) -> int:
    """Verify every instance file in ``--dir`` against ``--manifest``."""
    manifest = read_manifest(Path(args.manifest))
    files = _instance_files(Path(args.dir))
    mismatched = verify_manifest(manifest, files)

    if mismatched:
        joined = ", ".join(sorted(mismatched))
        print(f"Manifest verification FAILED for: {joined}")
        return 1

    print(f"Manifest OK: {len(files)} instance(s) verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
