#!/usr/bin/env bash
#
# Run a single Lethe configuration and stream its output to the terminal.
#
#   bash test_launch.sh                       # default config
#   bash test_launch.sh <path/to/config.jsonc>
#
set -euo pipefail

DEFAULT_CONFIG="configs/benchmark/lethe/AmazonPhotos_GCN_hard.jsonc"
CONFIG="${1:-$DEFAULT_CONFIG}"

# Always run from the repository root: configs reference paths such as
# resources/data and configs/snippets relative to it.
cd "$(dirname "$0")"

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG" >&2
    echo "Available benchmark configs:" >&2
    ls configs/benchmark/lethe/*.jsonc 2>/dev/null | head -5 >&2
    echo "  ..." >&2
    exit 1
fi

echo "=============================================================="
echo " Lethe -- running: $CONFIG"
echo "=============================================================="

python main.py "$CONFIG"

echo
echo "Done. Results were written to the path in the config's SaveValues measure"
echo "(usually under output/runs/)."
