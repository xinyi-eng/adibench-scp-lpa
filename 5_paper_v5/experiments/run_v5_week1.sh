#!/bin/bash
# run_v5_week1.sh - Week 1 v5 experiments on D:\dacd2026\5_paper_v5\
# Runs Table 7 (real few-shot), Table 1 (multi-seed), real t-SNE + grad norm.
# Estimated: ~10-12 GPU hours. Sequential (one process at a time).

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

echo "===== Week 1 v5: Table 7 few-shot (key cells) ====="
# 3 methods x 3 n_min (most informative) = 9 runs
for n in 5 50 500; do
  for m in ce dacd dacdpp; do
    GPU_WAIT
    echo "==== [$(date +%H:%M:%S)] fewshot $m n=$n ===="
    $PYTHON $RUNNER $m 100_1 42 1 \
      > $LOGS/fewshot_${m}_n${n}.log 2>&1 || echo "  FAILED: $m n=$n"
  done
done

echo "===== Week 1 v5: Table 1 multi-seed (LDAM/LA/ReCL) ====="
for m in ldam la recl; do
  for s in 0 7; do
    GPU_WAIT
    echo "==== [$(date +%H:%M:%S)] $m seed=$s 100:1 ===="
    $PYTHON $RUNNER $m 100_1 $s 3 \
      > $LOGS/${m}_100_1_seed${s}.log 2>&1 || echo "  FAILED: $m seed=$s"
  done
done

echo "===== Week 1 v5: Real t-SNE + grad norm ====="
GPU_WAIT
$PYTHON experiments/feats_extract.py ce 100_1 42 3 > $LOGS/feats_ce.log 2>&1 || echo "feats failed"
$PYTHON experiments/render_tsne_real.py > $LOGS/render_tsne.log 2>&1 || echo "tsne render failed"
$PYTHON experiments/render_gradnorm_real.py > $LOGS/render_gradnorm.log 2>&1 || echo "gradnorm failed"

echo "===== Week 1 v5 DONE: $(date) ====="
ls -la $RESULTS/*.json | wc -l
echo "results saved in $RESULTS"
