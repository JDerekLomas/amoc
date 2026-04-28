#!/bin/bash
# Capture model state from multiple Vercel deployments for comparison.
#
# Usage:
#   ./scripts/capture-versions.sh           # capture all versions in versions.json
#   ./scripts/capture-versions.sh 25000     # custom step count
#
# Each version runs in headless Chrome, simulates N steps from cold start,
# then dumps SST, diagnostics, and a screenshot to notebooks/runs/.
#
# The notebook (notebooks/amoc-analysis.ipynb) loads these files for comparison.

STEPS=${1:-25000}
DIR="notebooks/runs"
mkdir -p "$DIR"

echo "=== Capturing all versions at $STEPS steps ==="

# Read URLs from versions.json
URLS=$(node -e "
  const v = require('./v4-physics/versions.json');
  v.forEach(x => console.log(x.url));
")

for url in $URLS; do
  hash=$(echo "$url" | grep -o 'amoc-[a-z0-9]*' | sed 's/amoc-//')
  outfile="$DIR/${hash}-${STEPS}.json"

  if [ -f "$outfile" ]; then
    echo "Skip $hash (already captured)"
    continue
  fi

  echo ""
  echo "--- $hash ---"
  node scripts/capture-sim.mjs "$url" "$STEPS" "$DIR"
done

echo ""
echo "=== Done. Captures in $DIR/ ==="
ls -lh "$DIR"/*.json 2>/dev/null | awk '{print $5, $9}'
