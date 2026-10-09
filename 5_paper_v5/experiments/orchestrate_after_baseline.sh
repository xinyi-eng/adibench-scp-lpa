#!/usr/bin/env bash
# Run AFTER fill_pending_cells.sh finishes. Sequentially:
#   - Phase C: AML alpha sweep on dacd at 100:1 and 1000:1 (4 alphas each)
#   - Phase D: DACD++ few-shot sweep at 100:1 (5 n_min values, 3 methods)
#   - Phase E: regenerate figures + finalize paper + recompile

set -u
cd "$(dirname "$0")"
LOG=/d/dacd2026/3_experiments/v2_runs/orchestrate_after.log
echo "=== orchestrate_after start $(date) ===" > "$LOG"

# Wait for fill_pending to finish. The fill_pending_cells.sh creates a
# fill_pending.LOCK on start (manually retroactively for older versions)
# and fill_pending.DONE on completion. We accept either condition:
#   - DONE marker exists -> fill_pending finished
#   - All expected JSONs exist -> fill_pending finished (older scripts)
#   - LOCK file disappeared -> fill_pending finished and cleaned up
#   - Otherwise -> wait
expected_jsons=(
  "/d/dacd2026/3_experiments/v2_runs/ldam_100_1_seed7.json"
  "/d/dacd2026/3_experiments/v2_runs/la_100_1_seed7.json"
  "/d/dacd2026/3_experiments/v2_runs/recl_100_1_seed7.json"
  "/d/dacd2026/3_experiments/v2_runs/ce_7000_1_seed42.json"
  "/d/dacd2026/3_experiments/v2_runs/focal_7000_1_seed42.json"
  "/d/dacd2026/3_experiments/v2_runs/cb_7000_1_seed42.json"
  "/d/dacd2026/3_experiments/v2_runs/recl_7000_1_seed42.json"
)

wait_for_fill_pending() {
  local DONE_MARKER=/d/dacd2026/3_experiments/v2_runs/fill_pending.DONE
  local LOCK_FILE=/d/dacd2026/3_experiments/v2_runs/fill_pending.LOCK
  local last_log_size=-1
  while true; do
    if [ -f "$DONE_MARKER" ]; then
      echo "[$(date +%H:%M:%S)] fill_pending.DONE marker present; proceeding" | tee -a "$LOG"
      return 0
    fi
    if [ ! -f "$LOCK_FILE" ]; then
      echo "[$(date +%H:%M:%S)] fill_pending.LOCK gone (script exited); proceeding" | tee -a "$LOG"
      return 0
    fi
    # Check if all expected JSONs exist (older scripts skip the DONE marker)
    local all_present=1
    for f in "${expected_jsons[@]}"; do
      if [ ! -f "$f" ]; then
        all_present=0; break
      fi
    done
    if [ $all_present -eq 1 ]; then
      echo "[$(date +%H:%M:%S)] all expected JSONs exist; proceeding" | tee -a "$LOG"
      return 0
    fi
    # Also: check if fill_pending log has been quiet for 3 minutes (size unchanged)
    local cur=$(stat -c %s /d/dacd2026/3_experiments/v2_runs/fill_pending.log 2>/dev/null || echo 0)
    if [ "$cur" = "$last_log_size" ] && [ "$cur" -gt 100 ]; then
      echo "[$(date +%H:%M:%S)] fill_pending log size $cur unchanged twice; assuming done" | tee -a "$LOG"
      # Don't return yet; double-check
    fi
    last_log_size=$cur
    echo "[$(date +%H:%M:%S)] waiting for fill_pending... (done_count=$(ls /d/dacd2026/3_experiments/v2_runs/*.json 2>/dev/null | wc -l))" | tee -a "$LOG"
    sleep 90
  done
}

wait_for_fill_pending
touch /d/dacd2026/3_experiments/v2_runs/fill_pending.DONE  # write the marker retroactively so downstream scripts see it

run() {
  local method="$1" ratio="$2" seed="$3" extra="${4:-}"
  local out="/d/dacd2026/3_experiments/v2_runs/${method}_${ratio}_seed${seed}.json"
  if [ -f "$out" ] && [ "$method" != "dacd_aml" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $method $ratio seed=$seed (exists)" | tee -a "$LOG"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START $method $ratio seed=$seed $extra" | tee -a "$LOG"
  python3 multi_seed_runner_aml.py "$method" "$ratio" "$seed" 1 $extra >> "$LOG" 2>&1
  rm -f /d/dacd2026/3_experiments/v2_runs/${method}_${ratio}_seed99.json
  echo "[$(date +%H:%M:%S)] DONE  $method $ratio seed=$seed $extra" | tee -a "$LOG"
}

# --- C. AML alpha sweep ---
# alpha=0 means "no AML" -- gives the v3 fixed-beta baseline (already in v2_runs as dacd_*)
# We sweep alpha in {0.5, 1.0, 2.0} at 100:1 (seed=42) and 1000:1 (seed=42).
for ratio in 100_1 1000_1; do
  for alpha in 0.5 1.0 2.0; do
    run dacd_aml "$ratio" 42 "aml_alpha=${alpha}"
  done
done

# --- D. DACD++ few-shot sweep ---
# 5 n_min values; 3 methods (CE, DACD, DACD++) for each
# n_min=5 trains very fast (extremely few minor samples). n_min=200 trains ~3-5 min.
for n in 5 20 50 100 200; do
  for method in ce dacd dacdpp; do
    out="/d/dacd2026/3_experiments/v2_runs/${method}_fewshot_n${n}_seed42.json"
    if [ -f "$out" ]; then
      echo "[$(date +%H:%M:%S)] SKIP FEW ${method} n=${n} (exists)" | tee -a "$LOG"
      continue
    fi
    echo "[$(date +%H:%M:%S)] START FEW ${method} n=${n}" | tee -a "$LOG"
    python3 multi_seed_runner_fewshot.py "$method" 100_1 42 "$n" 1 >> "$LOG" 2>&1
    echo "[$(date +%H:%M:%S)] DONE  FEW ${method} n=${n}" | tee -a "$LOG"
  done
done

# --- E. Analysis ---
echo "[$(date +%H:%M:%S)] generating analysis/significance.md" | tee -a "$LOG"
python3 significance.py --v4-runs /d/dacd2026/3_experiments/v2_runs \
                        --out analysis/significance.md 2>> "$LOG"

echo "[$(date +%H:%M:%S)] regenerating paper figures" | tee -a "$LOG"
python3 generate_figures_v4.py 2>> "$LOG" || true
cp figs/*.png ../_compare_src/figs/ 2>> "$LOG"

echo "[$(date +%H:%M:%S)] recompiling revised_dacd.pdf" | tee -a "$LOG"
cd ../_compare_src && pdflatex -interaction=nonstopmode revision_paper.tex >>/dev/null 2>&1
pdflatex -interaction=nonstopmode revision_paper.tex >>/dev/null 2>&1
cp revision_paper.pdf ../revised_dacd.pdf
echo "[$(date +%H:%M:%S)] reproduce v3 dacd_bench.pdf from paper_v2.tex" | tee -a "$LOG"
python3 finalize_v2.py 2>> "$LOG" || true
cd ..

echo "=== orchestrate_after end $(date) ===" | tee -a "$LOG"
touch /d/dacd2026/3_experiments/v2_runs/orchestrate_after.DONE
