#!/bin/bash
# run_v5_week2.sh - Week 2 v5 experiments
# Runs v5 innovations: l2c, ami, arc, dacdv5 + MARBERTv2 baseline
# Estimated: ~12-15 GPU hours

set -e
cd /d/dacd2026/5_paper_v5
RESULTS=D:/dacd2026/5_paper_v5/results
LOGS=D:/dacd2026/5_paper_v5/experiments/logs
mkdir -p $RESULTS $LOGS
GPU_WAIT() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" -gt 1500 ]; do
    echo "$(date +%H:%M:%S) GPU busy, waiting..."
    sleep 30
  done
}

PYTHON="python -u"
RUNNER=experiments/multi_seed_runner_v5.py

echo "===== Week 2 v5: MARBERTv2 baseline (3 ratio x 1 seed) ====="
for r in 100_1 1000_1 7000_1; do
  GPU_WAIT
  echo "==== [$(date +%H:%M:%S)] marbert $r ===="
  $PYTHON $RUNNER marbert $r 42 1 > $LOGS/marbert_${r}.log 2>&1 || echo "  FAILED: marbert $r"
done

echo "===== Week 2 v5: 4 innovations (l2c/ami/arc/dacdv5) on 100:1 ====="
for m in l2c ami arc dacdv5; do
  GPU_WAIT
  echo "==== [$(date +%H:%M:%S)] $m 100:1 ===="
  $PYTHON $RUNNER $m 100_1 42 3 > $LOGS/${m}_100_1.log 2>&1 || echo "  FAILED: $m"
done

echo "===== Week 2 v5: DACD v5 on 1000:1 and 7000:1 ====="
for r in 1000_1 7000_1; do
  GPU_WAIT
  echo "==== [$(date +%H:%M:%S)] dacdv5 $r ===="
  $PYTHON $RUNNER dacdv5 $r 42 1 > $LOGS/dacdv5_${r}.log 2>&1 || echo "  FAILED: dacdv5 $r"
done

echo "===== Week 2 v5: CEDA cross-encoder (slow due to dual encoder) ====="
GPU_WAIT
echo "==== [$(date +%H:%M:%S)] ceda 100:1 ===="
$PYTHON $RUNNER ceda 100_1 42 1 > $LOGS/ceda_100_1.log 2>&1 || echo "  CEDA may OOM (8GB); fallback later"

echo "===== Week 2 v5 DONE: $(date) ====="
ls -la $RESULTS/*.json | wc -l
echo "results saved in $RESULTS"
