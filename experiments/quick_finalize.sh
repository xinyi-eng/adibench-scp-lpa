#!/usr/bin/env bash
# Quick finalize: as soon as fill_pending_cells finishes, regenerate figures
# and paper tables WITHOUT waiting for AML/few-shot sweeps. Frees up dacd_bench.pdf
# and revised_dacd.pdf updates 3-7 hours earlier than orchestrate_after.

set -u
cd "$(dirname "$0")"
LOG=/d/dacd2026/3_experiments/v2_runs/quick_finalize.log
echo "=== quick_finalize start $(date) ===" > "$LOG"

LOCK_FILE=/d/dacd2026/3_experiments/v2_runs/fill_pending.LOCK
EXPECTED_DIR=/d/dacd2026/3_experiments/v2_runs

# Wait until fill_pending is done. Detect via:
#   - LOCK removed (clean exit) OR
#   - All expected JSONs present
expected_jsons=(
  "$EXPECTED_DIR/ldam_100_1_seed7.json"
  "$EXPECTED_DIR/la_100_1_seed7.json"
  "$EXPECTED_DIR/recl_100_1_seed7.json"
  "$EXPECTED_DIR/ce_7000_1_seed42.json"
  "$EXPECTED_DIR/focal_7000_1_seed42.json"
  "$EXPECTED_DIR/cb_7000_1_seed42.json"
  "$EXPECTED_DIR/recl_7000_1_seed42.json"
)

while true; do
  if [ ! -f "$LOCK_FILE" ]; then
    echo "[$(date +%H:%M:%S)] LOCK gone; fill_pending done" | tee -a "$LOG"
    break
  fi
  all_present=1
  for f in "${expected_jsons[@]}"; do
    if [ ! -f "$f" ]; then all_present=0; break; fi
  done
  if [ $all_present -eq 1 ]; then
    echo "[$(date +%H:%M:%S)] all expected JSONs present; fill_pending done" | tee -a "$LOG"
    break
  fi
  echo "[$(date +%H:%M:%S)] waiting for fill_pending... (jsons=$(ls $EXPECTED_DIR/*.json 2>/dev/null | wc -l))" | tee -a "$LOG"
  sleep 120
done

# Run finalize + figure regen + recompile
cd ..
echo "[$(date +%H:%M:%S)] generating figures" | tee -a "$LOG"
cd experiments
python3 generate_figures_v4.py 2>> "$LOG"
cp -f figs/*.png ../_compare_src/figs/ 2>> "$LOG"

echo "[$(date +%H:%M:%S)] finalizing paper_v2.tex + dacd_bench.pdf" | tee -a "$LOG"
cd experiments
python3 finalize_v2.py 2>> "$LOG" || true

echo "[$(date +%H:%M:%S)] recompiling revised_dacd.pdf" | tee -a "$LOG"
cd ../_compare_src
pdflatex -interaction=nonstopmode revision_paper.tex >>/dev/null 2>&1
pdflatex -interaction=nonstopmode revision_paper.tex >>/dev/null 2>&1
cp -f revision_paper.pdf ../revised_dacd.pdf
echo "[$(date +%H:%M:%S)] quick_finalize done" | tee -a "$LOG"
touch /d/dacd2026/3_experiments/v2_runs/quick_finalize.DONE
