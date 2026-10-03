"""Remove the duplicate fullft table that's pushing us over 8 pages."""
import re, io

P = r"C:/Users/zsndz/Desktop/dacd_bench_v4_20260826_153119/adibench_paper.tex"
with io.open(P, "r", encoding="utf-8") as f:
    s = f.read()

# Find the duplicate full-data finetune table block
start_marker = "\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{lcc}\n\\toprule\nEvaluation"
# We have multiple table* - find specifically the one with "Overall acc" "Macro F1"
start_idx = s.find(start_marker)
if start_idx == -1:
    print("Couldn't find table*")
else:
    end_idx = s.find("\\end{table*}", start_idx)
    end_idx += len("\\end{table*}")
    new_s = s[:start_idx] + s[end_idx:]
    # Also remove the now-empty section header that referenced it
    # Actually keep the section, but delete "FineTune (full 439K)" row in the robustness table
    # (we already did that).
    with io.open(P, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_s)
    print(f"Removed {end_idx - start_idx} chars")
    print("New file size:", len(new_s))