"""Rewrite C drive runner.py with full method support.

Replace the older runner.py (which only knows 9 methods) with the
newer runner from D drive (which knows all 13 methods).

Strategy: just copy D drive's runner.py over C drive.
"""
import shutil
import io

SRC = r"D:/dacd2026/adibench_v1/experiments/run_adibench_baseline.py"
DST = r"C:/Users/zsndz/Desktop/dacd_bench_v4_20260826_153119/experiments/run_adibench_baseline.py"

# Read D drive runner
with io.open(SRC, "r", encoding="utf-8") as f:
    src = f.read()

# Check it has softanchor
assert "SoftAnchorProtoNet" in src, "D drive runner missing SoftAnchor"
assert "lpa_protonet_step" in src, "D drive runner missing lpa step"

# Write to C drive
with io.open(DST, "w", encoding="utf-8", newline="\n") as f:
    f.write(src)
print(f"Copied {SRC} -> {DST} ({len(src)} bytes)")