"""Fix: move SoftAnchorProtoNet + softanchor_step BEFORE ALL_BASELINES dict."""
import io

BASE = r"D:/dacd2026/adibench_v1/adibench/baselines.py"


def read(p):
    with io.open(p, "r", encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


src = read(BASE)

# Find the SoftAnchor class block (everything from "class SoftAnchorProtoNet" to
# "return supcon + ce_weight * ce + anchor_gamma * anc, supcon.item(), ce.item(), anc.item()")
# and move it to just before "ALL_BASELINES = {"
import re
# Match from "class SoftAnchorProtoNet" up to the end of softanchor_step function
match = re.search(
    r"\n\nclass SoftAnchorProtoNet\(nn\.Module\):.*?return supcon \+ ce_weight \* ce \+ anchor_gamma \* anc, supcon\.item\(\), ce\.item\(\), anc\.item\(\)\n",
    src, re.DOTALL,
)
if not match:
    print("Could not find SoftAnchor block")
else:
    block = match.group(0)
    src = src.replace(block, "\n", 1)  # Remove from current location
    # Insert before ALL_BASELINES
    src = src.replace("ALL_BASELINES = {", block.lstrip() + "\n\nALL_BASELINES = {", 1)
    print("Moved SoftAnchorProtoNet before ALL_BASELINES")
    write(BASE, src)