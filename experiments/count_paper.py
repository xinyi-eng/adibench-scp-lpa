"""Count paper structure elements."""
import re
with open(r"C:\Users\zsndz\Desktop\dacd_bench_v4_20260826_153119\adibench_paper.tex") as f:
    s = f.read()
print("Sections:", len(re.findall(r"\\section", s)))
print("Subsections:", len(re.findall(r"\\subsection", s)))
print("Figures:", len(re.findall(r"\\begin\{figure", s)))
print("Tables:", len(re.findall(r"\\begin\{table", s)))
print("Figure* (wide):", len(re.findall(r"\\begin\{figure\*", s)))
print("Table* (wide):", len(re.findall(r"\\begin\{table\*", s)))