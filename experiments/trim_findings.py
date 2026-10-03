"""Remove Finding 5 (robustness) text to save space - it duplicates the Robustness section."""
import re, io

P = r"C:/Users/zsndz/Desktop/dacd_bench_v4_20260826_153119/adibench_paper.tex"
with io.open(P, "r", encoding="utf-8") as f:
    s = f.read()

# Remove Finding 5 (robustness checks - we have a separate section now)
start = s.find("\\textbf{Finding 6: Robustness checks confirm the win.}")
if start == -1:
    start = s.find("\\textbf{Finding 5: SCP and LPA both beat")
if start == -1:
    print("Not found")
else:
    # Find next Finding or end of Findings section
    end = s.find("\\textbf{Finding 4:", start)
    if end == -1:
        end = s.find("\\section{", start)
    if end > start:
        removed = s[start:end]
        s = s[:start] + s[end:]
        # Also remove the orphaned "**Finding 4: ..." since it'll be the wrong number
        # Just rename F4 -> F5 etc, or remove
        with io.open(P, "w", encoding="utf-8", newline="\n") as f:
            f.write(s)
        print(f"Removed {end - start} chars")

# Also need to ensure the conclusion gets in. Let me see what's at the very end.