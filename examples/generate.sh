#!/usr/bin/env bash
#
# Regenerate the example dataset under examples/data.
#
# Generation is fully deterministic (driven by an explicit seed), so re-running
# this script yields byte-identical files. The companion test
# (tests/test_examples.py) regenerates these instances and fails if the
# committed files drift from what the current generator produces.
#
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p data

pisinger-knapsack generate --n 20   --correlation uncorrelated --seed 1 --out data/n20_uncorrelated.json
pisinger-knapsack generate --n 20   --correlation weakly       --seed 2 --out data/n20_weakly.json
pisinger-knapsack generate --n 20   --correlation strongly     --seed 3 --out data/n20_strongly.json
pisinger-knapsack generate --n 1000 --correlation uncorrelated --seed 4 --out data/n1000_uncorrelated.json
pisinger-knapsack generate --n 1000 --correlation weakly       --seed 5 --out data/n1000_weakly.json
pisinger-knapsack generate --n 1000 --correlation strongly     --seed 6 --out data/n1000_strongly.json

pisinger-knapsack manifest build --dir data --out data/manifest.json
