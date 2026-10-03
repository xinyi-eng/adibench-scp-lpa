#!/usr/bin/env bash
# Auto-loop: every ~10 min, run finalize_v2.py to recompile paper.
# Stops gracefully when launch_v2.sh finishes (no new JSONs for 60 min).
# Logs to v2_runs/auto.log.

cd /c/Users/zsndz/Desktop/论文/experiments

LOG=/d/dacd2026/3_experiments/v2_runs/auto.log
V2=/d/dacd2026/3_experiments/v2_runs
echo "=== auto_finalize start $(date) ===" > "$LOG"

prev_count=0
idle_runs=0

while true; do
  count=$(ls "$V2"/*.json 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] check: json count=$count (prev=$prev_count) idle=$idle_runs" >> "$LOG"

  if [ "$count" -gt "$prev_count" ]; then
    echo "[$(date +%H:%M:%S)] new JSONs detected, running finalize_v2.py" >> "$LOG"
    python3 finalize_v2.py >> "$LOG" 2>&1
    prev_count=$count
    idle_runs=0
  else
    idle_runs=$((idle_runs + 1))
    if [ "$idle_runs" -ge 6 ]; then
      # 60 min no new JSONs -> assume experiments done; one final pass + exit
      echo "[$(date +%H:%M:%S)] 60min idle, finalizing and exiting" >> "$LOG"
      python3 finalize_v2.py >> "$LOG" 2>&1
      break
    fi
  fi

  sleep 600  # 10 min
done

echo "=== auto_finalize end $(date) ===" >> "$LOG"