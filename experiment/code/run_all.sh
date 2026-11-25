#!/usr/bin/env bash
# One-command rebuild of the NORS hospitalization-ranking analysis from the raw snapshot.
#   raw snapshot -> prepare_data.py -> analyze.py (+ bootstrap & stacked M-estimation & VoI)
#   -> compare_results.py -> check_tables.py
#
# Usage: bash code/run_all.sh [EXP_DIR] [OUT_DIR] [B]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "${1:-$HERE/..}" && pwd)"
OUT="${2:-$EXP/rerun}"
B="${3:-2000}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
if [ "$OUT" = "$EXP" ]; then echo "OUT_DIR must differ from EXP_DIR" >&2; exit 2; fi
RAW="$EXP/data/raw/nors_soda_full_20261125T040830Z.csv"
META="$EXP/data/raw/nors_soda_full_20261125T040830Z.json"
PRESERVED="$EXP/analysis/results.json"
mkdir -p "$OUT/data/derived" "$OUT/rebuild_check" "$OUT/logs"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

echo "[$(stamp)] prepare_data" | tee "$OUT/logs/run_all.log"
python3 "$HERE/prepare_data.py" --raw "$RAW" --meta "$META" --out "$OUT/data/derived" 2>&1 | tee "$OUT/logs/prepare_data.log"

echo "[$(stamp)] analyze (bootstrap B=$B, 2 workers)" | tee -a "$OUT/logs/run_all.log"
python3 "$HERE/analyze.py" --derived "$OUT/data/derived" --out "$OUT/rebuild_check/results_rebuilt.json" \
  --bootstrap-B "$B" --workers 2 --bootstrap-jsonl "$OUT/logs/bootstrap_replicates.jsonl" 2>&1 | tee "$OUT/logs/analyze.log"

echo "[$(stamp)] compare with the canonical results" | tee -a "$OUT/logs/run_all.log"
python3 "$HERE/compare_results.py" --preserved "$PRESERVED" --rebuilt "$OUT/rebuild_check/results_rebuilt.json" \
  --rebuilt-jsonl "$OUT/logs/bootstrap_replicates.jsonl" \
  --out-json "$OUT/rebuild_check/rebuild_check.json" --out-md "$OUT/rebuild_check/rebuild_check.md" 2>&1 | tee "$OUT/logs/compare.log"

if [ -f "$HERE/make_outputs.py" ] && [ -d "$EXP/data/analysis_tables" ]; then
  echo "[$(stamp)] render tables from the rebuilt results and compare with the delivered tables" | tee -a "$OUT/logs/run_all.log"
  python3 "$HERE/check_tables.py" --rebuilt "$OUT/rebuild_check/results_rebuilt.json" \
    --reference-tables "$EXP/data/analysis_tables" --out-json "$OUT/rebuild_check/table_check.json" 2>&1 | tee -a "$OUT/logs/check_tables.log"
fi
echo "[$(stamp)] done; report: $OUT/rebuild_check/rebuild_check.md" | tee -a "$OUT/logs/run_all.log"
