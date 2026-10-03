#!/usr/bin/env bash
# Fill the [pending multi-seed] cells in paper_v2.tex tables.
# Targets:
#   100:1 ldam/la/recl  -- 3 methods x seeds {0, 7}        = 6 runs (1 epoch, ~12 min ea)
#   1000:1 ce/focal/cb   -- 3 methods x seed 42            = 3 runs (1 epoch,  ~3 min ea)
#   7000:1 ce/focal/cb/recl -- 4 methods x seed 42         = 4 runs (1 epoch, ~6 sec ea)
# Total wall clock:  ~1.5 hours
#
# Each run writes /d/dacd2026/3_experiments/v2_runs/<method>_<ratio>_seed<N>.json
# which the finalize_v2.py + significance.py scripts consume.
#
# Run from inside `experiments/`.

set -u
cd "$(dirname "$0")"
LOG=/d/dacd2026/3_experiments/v2_runs/fill_pending.log
DONE_MARKER=/d/dacd2026/3_experiments/v2_runs/fill_pending.DONE
LOCK_FILE=/d/dacd2026/3_experiments/v2_runs/fill_pending.LOCK

# Don't run twice
if [ -f "$DONE_MARKER" ]; then
  echo "fill_pending.DONE marker exists; exiting"
  exit 0
fi
rm -f /d/dacd2026/3_experiments/v2_runs/dacd_aml_7000_1_seed99.json \
      /d/dacd2026/3_experiments/v2_runs/dacd_aml_100_1_seed99.json 2>/dev/null

echo "=== fill_pending start $(date) ===" > "$LOG"
touch "$LOCK_FILE"

run() {
  local method="$1" ratio="$2" seed="$3" extra="${4:-}"
  local out="/d/dacd2026/3_experiments/v2_runs/${method}_${ratio}_seed${seed}.json"
  if [ -f "$out" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $method $ratio seed=$seed (already exists)" | tee -a "$LOG"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START $method $ratio seed=$seed" | tee -a "$LOG"
  if [ -n "$extra" ]; then
    python3 -u multi_seed_runner.py "$method" "$ratio" "$seed" 1 "dacd_alpha=$extra" >> "$LOG" 2>&1
  else
    python3 -u multi_seed_runner.py "$method" "$ratio" "$seed" 1 >> "$LOG" 2>&1
  fi
  echo "[$(date +%H:%M:%S)] DONE  $method $ratio seed=$seed" | tee -a "$LOG"
  # Pause to let GPU memory release
  sleep 5
}

# --- A. 100:1 baseline fills (3 methods x 2 seeds = 6 runs) ---
for method in ldam la recl; do
  for seed in 0 7; do
    run "$method" 100_1 "$seed"
  done
done

# --- B. 1000:1 fills for ce/focal/cb (3 runs) ---
for method in ce focal cb; do
  run "$method" 1000_1 42
done

# --- C. 7000:1 fills for ce/focal/cb/recl (4 runs) ---
for method in ce focal cb recl; do
  run "$method" 7000_1 42
done

# --- D. ReCL 7000:1 (was missing) ---
# run recl 7000_1 42  -- already done in v2_runs list above

echo "=== fill_pending end $(date) ===" | tee -a "$LOG"
touch /d/dacd2026/3_experiments/v2_runs/fill_pending.DONE
ls -la /d/dacd2026/3_experiments/v2_runs/ | wc -l
