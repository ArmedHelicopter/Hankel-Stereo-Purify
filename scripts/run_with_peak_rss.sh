#!/usr/bin/env bash
# Run the CLI under GNU time -v to record Max resident set size (NF-01 / week 3).
# Usage from repo root:
#   ./scripts/run_with_peak_rss.sh in.flac out.flac -- --window-length 256 --rank 64
# Arguments after -- are passed to python -m src.cli

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  echo "Usage: $0 <input.flac> <output.flac> [-- extra cli args...]"
  echo "Requires GNU time (/usr/bin/time) for Max resident set size."
  exit 0
fi

IN="${1:?input flac}"
OUT="${2:?output flac}"
shift 2
if [[ "${1:-}" == "--" ]]; then
  shift
fi

if ! command -v /usr/bin/time >/dev/null 2>&1; then
  echo "GNU time not found; running without -v." >&2
  exec python -m src.cli "$IN" "$OUT" "$@"
fi

exec /usr/bin/time -v python -m src.cli "$IN" "$OUT" "$@"
