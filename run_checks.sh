#!/usr/bin/env bash
set -euo pipefail
python -m unittest discover -s tests -v
python -m multicalib --config configs/example.yaml simulate
