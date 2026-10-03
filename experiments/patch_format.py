"""Fix table overflows and equation overflows in adibench_paper.tex."""
import io

PATH = r"C:/Users/zsndz/Desktop/dacd_bench_v4_20260826_153119/adibench_paper.tex"


def read():
    with io.open(PATH, "r", encoding="utf-8") as f:
        return f.read()


def write(s):
    with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


src = read()

# --- Equation overflow (SupCon) ---
old_eq = """\\begin{equation}
  \\mathcal{L}_{\\text{SupCon}} = -\\frac{1}{N}\\sum_{i=1}^N
  \\frac{1}{|P(i)|}\\sum_{p\\in P(i)}
  \\log\\frac{\\exp(\\mathrm{sim}(z_i,z_p)/\\tau)}{\\sum_{a\\in A(i)}\\exp(\\mathrm{sim}(z_i,z_a)/\\tau)},
\\end{equation}"""
new_eq = """\\begin{equation}
  \\small
  \\mathcal{L}_{\\text{SupCon}} = -\\frac{1}{N}\\sum_{i=1}^N
  \\frac{1}{|P(i)|}\\sum_{p\\in P(i)}
  \\log\\frac{\\exp(\\mathrm{sim}(z_i,z_p)/\\tau)}{\\sum_{a\\in A(i)}\\exp(\\mathrm{sim}(z_i,z_a)/\\tau)}\\!,
\\end{equation}"""
assert src.count(old_eq) == 1, f"eq count {src.count(old_eq)}"
src = src.replace(old_eq, new_eq)
print("ok equation SupCon")

# --- Convert overflow tables from [t] to table* ---
# Lines: 477-492 (sig test 5 cols), 498-513 (multiseed 7 cols),
#         519-534 (scp_multiseed 7 cols), 540-555 (finetune_multiseed 7 cols),
#         579-602 (ablation 7 cols)
# Pattern: "\begin{table}[t]\n\centering" preceded by something that includes
# 6+ column tabular (lccccccc, lccccccc, etc.). Simpler: every small tabular
# with 6+ columns should be table*.
import re

def to_tablestar(src_text, start_pattern):
    """Convert \begin{table}[t] -> \begin{table*}[t] and matching \end{table} -> \end{table*}."""
    n = src_text.count(start_pattern)
    if n == 0:
        return src_text
    src_text = src_text.replace(start_pattern, start_pattern.replace("[t]", "[t]").replace("\\begin{table}", "\\begin{table*}"))
    return src_text


# Instead of regex magic, do targeted replacements. Tables we need to convert:
patterns = [
    "\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lcccc}\n\\toprule\nCell & ProtoNet & MatchingNet",  # sig test
    "\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lccccccc}\n\\toprule\nCell & s=0",  # multiseed
    "\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lccccccc}\n\\toprule\nCell & s=0 & s=7 & s=42 & s=123 & mean$\\pm$std",  # scp_multiseed
    "\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lccccccc}\n\\toprule\nCell & s=0 & s=7 & s=42 & s=123 & mean$\\pm$std",  # finetune_multiseed
    "\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lcccccc}\n\\toprule\n& \\multicolumn{2}{c}{NADI 18} & \\multicolumn{2}{c}{NADI 5} & \\multicolumn{2}{c}{amgadhasan 5}",  # ablation
]
for p in patterns:
    n = src.count(p)
    print(f"pattern matches: {n}  ({p[:50]}...)")

# Simpler approach: convert all overflow tables via regex.
# Match \begin{table}[t]\n\centering (any)\n (any)\n\begin{tabular}{...} with 6+ columns
pattern = re.compile(
    r"\\begin\{table\}\[t\]\n\\centering(?:\\small)?\n\\begin\{tabular\}\{([lcrd ]+)\}\n"
)

def repl(m):
    cols = m.group(1)
    if len(cols) >= 6:
        return m.group(0).replace("\\begin{table}[t]", "\\begin{table*}[t]")
    return m.group(0)


new_src, n_conv = pattern.subn(repl, src)
print(f"converted {n_conv} tables to table*")
src = new_src

# Now also need to close those table* with \end{table*} instead of \end{table}.
# Find pairs. Easier: any \end{table} preceded by a tabular with 6+ cols inside
# already-converted table*. Let me just count \begin{table*} and \end{table}
n_open_star = src.count("\\begin{table*}")
n_close = src.count("\\end{table}")
n_open = src.count("\\begin{table}[")
print(f"table* open={n_open_star}, [t] open={n_open}, \\end{{table}} total={n_close}")

# State machine: reset on EITHER \end{table} OR \end{table*}.
out = []
in_star = False
for line in src.split("\n"):
    s = line.strip()
    if s == "\\begin{table*}[t]":
        in_star = True
        out.append(line)
    elif s in ("\\end{table}", "\\end{table*}"):
        # If we're in a table* and we see \end{table}, convert.
        if s == "\\end{table}" and in_star:
            out.append(line.replace("\\end{table}", "\\end{table*}"))
        else:
            out.append(line)
        # Reset on EITHER close.
        in_star = False
    else:
        out.append(line)
src = "\n".join(out)
print("balanced table*/table")
write(src)
print("patched")