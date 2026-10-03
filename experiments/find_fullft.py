"""Find and remove the fullft table block (the one with Evaluation/Metric/Score)."""
import re, io

P = r"C:/Users/zsndz/Desktop/dacd_bench_v4_20260826_153119/adibench_paper.tex"
with io.open(P, "r", encoding="utf-8") as f:
    s = f.read()

# Find all \begin{table*}...\end{table*} blocks
pattern = re.compile(r"\\begin\{table\*\}\[t\]\n.*?\\end\{table\*\}", re.DOTALL)
matches = list(pattern.finditer(s))
print(f"Found {len(matches)} table* blocks")
for i, m in enumerate(matches):
    # Find which has Evaluation/Metric/Score
    if "Evaluation" in m.group() and "Macro F1" in m.group():
        print(f"  Match {i} (chars {m.start()}-{m.end()}): contains fullft")
        # Remove it
        s = s[:m.start()] + s[m.end():]
        print(f"  Removed {m.end() - m.start()} chars")
        break

# Also remove the "Figure:" caption if duplicated and any orphaned references
with io.open(P, "w", encoding="utf-8", newline="\n") as f:
    f.write(s)
print(f"New file size: {len(s)}")