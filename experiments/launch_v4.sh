#!/usr/bin/env bash
# Phase A: 100:1 full sweep -- 8 methods x 3 seeds x 3 epochs (~12 hours)
# Phase B: 1000:1 and 7000:1 baselines at seed 42 (~30 min)
# Phase C: AML-DeBias alpha sweep at 100:1 and 1000:1 (~6 hours)
# Phase D: DACD++ few-shot sweep at 100:1 with n_min \in {5,20,50,100,200} (~6 hours)
# All outputs land in ../v4_runs/<method>_<ratio>_seed<seed>.json

set -u
cd "$(dirname "$0")"

LOG=../v4_runs/launch.log
mkdir -p ../v4_runs
echo "=== launch_v4 start $(date) ===" > "$LOG"

# Phase A: 100:1 multi-seed (3 epochs each, 8 methods x 3 seeds = 24 runs)
for method in ce focal cb dacd dacdpp ldam la recl; do
  for seed in 0 42 7; do
    echo "[$(date +%H:%M:%S)] Phase A: $method 100_1 seed=$seed 3ep" >> "$LOG"
    python3 multi_seed_runner.py "$method" 100_1 "$seed" 3 >> "$LOG" 2>&1
  done
done

# Phase B: 1000:1 and 7000:1 seed-42 (1 epoch each, all 8 methods)
for ratio in 1000_1 7000_1; do
  for method in ce focal cb dacd dacdpp ldam la recl; do
    echo "[$(date +%H:%M:%S)] Phase B: $method $ratio 1ep seed=42" >> "$LOG"
    python3 multi_seed_runner.py "$method" "$ratio" 42 1 >> "$LOG" 2>&1
  done
done

# Phase C: AML-DeBias alpha sweep at 100:1 (default alpha=1.0; sweep other values)
for alpha in 0.0 0.5 1.0 2.0; do
  echo "[$(date +%H:%M:%S)] Phase C: alpha=$alpha dacd 100_1 seed=42 1ep" >> "$LOG"
  # alpha=0.0 fixed-beta fallback (no AMLDeBias); alpha=1.0 is the default
  python3 multi_seed_runner.py dacd 100_1 42 1 "dacd_alpha=${alpha}" >> "$LOG" 2>&1
done

# Phase D: DACD++ few-shot sweep at 100:1 base split
for n_min in 5 20 50 100 200; do
  echo "[$(date +%H:%M:%S)] Phase D: dacdpp few_shot n_min=$n_min seed=42 1ep" >> "$LOG"
  python3 multi_seed_runner.py dacdpp 100_1 42 1 "n_min=${n_min}" >> "$LOG" 2>&1
done

echo "=== launch_v4 end $(date) ===" >> "$LOG"

# Analysis
python3 significance.py --v4-runs ../v4_runs --out analysis/significance.md >> "$LOG" 2>&1
python3 generate_figures_v4.py >> "$LOG" 2>&1

# Re-compile paper (placeholder: user runs this manually after reviewing figures)
echo "Run: cd .. && pdflatex -interaction=nonstopmode _compare_src/revision_paper.tex" >> "$LOG"
echo "and copy: cp _compare_src/revision_paper.pdf revised_dacd.pdf" >> "$LOG"
