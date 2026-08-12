#!/usr/bin/env bash
# Run the Fleet Insights API locally.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
