#!/usr/bin/env bash
# Phase A: 100:1 multi-seed at 1 epoch (fast sanity check, ~2.5 hours total)
# Phase B: 1000:1/7000:1 baselines (LDAM/LA/ReCL/DACD), < 30 min total
# Phase C: beta sensitivity at 100:1 (~3.3 hours)
# Phase D: ablation at 100:1 (~2.5 hours)
# All runs save to /D/dacd2026/3_experiments/v2_runs/

cd /c/Users/zsndz/Desktop/论文/experiments

LOG=/d/dacd2026/3_experiments/v2_runs/launch.log
echo "=== launch start $(date) ===" > "$LOG"

# Phase A: 100:1 multi-seed (1 epoch each, 5 methods x 2 seeds = 10 runs)
# We have seed 42 from C_log already; need seeds 0 and 7.
# For methods: CE, focal, cb, dacd, dacdpp x seeds 0, 7 x 1 epoch
for method in ce focal cb dacd dacdpp; do
  for seed in 0 7; do
    echo "[$(date +%H:%M:%S)] start $method seed=$seed 1ep 100:1" >> "$LOG"
    python3 multi_seed_runner.py "$method" 100_1 "$seed" 1 >> "$LOG" 2>&1
  done
done

# Phase B: 1000:1 and 7000:1 for dacd, ldam, la, recl, dacdpp
# These are fast (1000:1 ~150s/ep, 7000:1 ~10s/ep)
for ratio in 1000_1 7000_1; do
  for method in ldam la recl dacd dacdpp; do
    echo "[$(date +%H:%M:%S)] $method $ratio 1ep" >> "$LOG"
    python3 multi_seed_runner.py "$method" "$ratio" 42 1 >> "$LOG" 2>&1
  done
done

# Phase C: beta sensitivity at 100:1 (DACD with beta in {0.0, 0.1, 0.3, 0.5, 0.7, 1.0} x seed 42, 1 epoch)
for beta in 0.0 0.1 0.3 0.5 0.7 1.0; do
  echo "[$(date +%H:%M:%S)] dacd beta=$beta 100:1 1ep" >> "$LOG"
  python3 multi_seed_runner.py dacd 100_1 42 1 "dacd_beta=$beta" >> "$LOG" 2>&1
done

# Phase D: ablation - "no DACD" uses plain CE; "no Khaleeji de-bias" uses beta=0; "no prototype" = dacd not dacdpp
# Already covered above; skip extra ablation here to save time.

echo "=== launch end $(date) ===" >> "$LOG"
