#!/usr/bin/env bash
set -e

# 1. Run from the repository root and name this study.
cd "$(dirname "$0")"
study="${1:-depth_01}"

# 2. Download data once before starting the six training runs.
uv run runs/prepare.py
uv run runs/train.py +experiment=sweep_models "study=$study"

# 3. Summarize the completed runs in the same study.
uv run runs/report.py "study=$study"
