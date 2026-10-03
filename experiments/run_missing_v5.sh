#!/bin/bash
# run_missing_v5.sh - complete all missing experiments sequentially
# Estimated: ~3-4 GPU hours total

set -e
cd /d/dacd2026/5_paper_v5
RESULTS=D:/dacd2026/5_paper_v5/results
LOGS=D:/dacd2026/5_paper_v5/experiments/logs
mkdir -p $LOGS

PYTHON="python -u"
RUNNER=experiments/multi_seed_runner_v5.py
PYTHONPATH=D:/dacd2026/5_paper_v5:D:/dacd2026/site-packages
export PYTHONPATH

run() {
  local method=$1; local ratio=$2; local seed=$3; local epochs=$4
  echo "==== [$(date +%H:%M:%S)] $method $ratio seed=$seed epochs=$epochs ===="
  $PYTHON $RUNNER $method $ratio $seed $epochs > $LOGS/missing_${method}_${ratio}_seed${seed}_ep${epochs}.log 2>&1 || echo "  FAILED: $method"
}

# Phase 1: 100:1 missing methods (1 epoch)
echo "===== Phase 1: 100:1 missing baselines ====="
run ami       100_1 42 1
run dacdpp    100_1 42 1
run marbert   100_1 42 1
run ldam      100_1 42 1
run recl      100_1 42 1

# Phase 2: 1000:1 missing methods
echo "===== Phase 2: 1000:1 missing methods ====="
run ami       1000_1 42 1
run arc       1000_1 42 1
run dacdpp    1000_1 42 1

# Phase 3: 7000:1 - quick runs
echo "===== Phase 3: 7000:1 quick ====="
run l2c       7000_1 42 1
run dacdv5    7000_1 42 1

# Phase 4: 3-epoch runs for fair comparison
echo "===== Phase 4: 3-epoch critical runs ====="
run ce        100_1 42 3
run l2c       100_1 42 3
run dacd      100_1 42 3
run dacdv5    100_1 42 3

echo "===== ALL MISSING RUNS COMPLETE: $(date) ====="
ls -la $RESULTS/*.json | wc -l
