#!/usr/bin/env bash
# Train CE and DACD for 1 epoch each on 100:1, dump last-layer features for t-SNE.
# Adds ~30 min after orchestrate_after finishes.

set -u
cd "$(dirname "$0")"
LOG=/d/dacd2026/3_experiments/v2_runs/features_extract.log
echo "=== features_extract start $(date) ===" > "$LOG"

# wait for orchestrate_after to write its DONE marker
DONE_MARKER=/d/dacd2026/3_experiments/v2_runs/orchestrate_after.DONE
LOCK_FILE=/d/dacd2026/3_experiments/v2_runs/orchestrate_after.LOCK
while [ ! -f "$DONE_MARKER" ]; do
  if [ ! -f "$LOCK_FILE" ]; then
    echo "[$(date +%H:%M:%S)] no lock or done marker; assuming orchestrate_after finished previously" | tee -a "$LOG"
    break
  fi
  echo "[$(date +%H:%M:%S)] waiting for orchestrate_after DONE marker..." | tee -a "$LOG"
  sleep 60
done

# Train CE on 100:1 and dump features
for method in ce dacd; do
  echo "[$(date +%H:%M:%S)] extracting features: $method 100:1" | tee -a "$LOG"
  python3 feats_extract.py $method 100_1 42 >> "$LOG" 2>&1
done

# Capture grad-norm trajectory for AML (single 100:1 run with log_grad=1)
echo "[$(date +%H:%M:%S)] capture grad-norm trajectory" | tee -a "$LOG"
OUT_GRAD=/d/dacd2026/3_experiments/v2_runs/dacd_aml_100_1_gradnorm_seed42.json
if [ ! -f "$OUT_GRAD" ]; then
  python3 multi_seed_runner_aml.py dacd_aml 100_1 42 1 aml_alpha=1.0 log_grad=1 >> "$LOG" 2>&1
fi
python3 render_gradnorm_real.py 2>> "$LOG" || true

# Render t-SNE
echo "[$(date +%H:%M:%S)] rendering t-SNE" | tee -a "$LOG"
python3 render_tsne_real.py 2>> "$LOG" || true

echo "=== features_extract end $(date) ===" | tee -a "$LOG"
