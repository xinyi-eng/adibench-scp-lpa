"""Fix runner.py to recognize softanchor + check current make_model state."""
import io, sys

RUNNER = r"D:/dacd2026/adibench_v1/experiments/run_adibench_baseline.py"


def read(p):
    with io.open(p, "r", encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


src = read(RUNNER)
# Print current make_model
import re
m = re.search(r"def make_model\(.*?\n(?=\ndef )", src, re.DOTALL)
if m:
    print("Current make_model:")
    print(m.group(0))

# Add softanchor handler if not present
if '"softanchor"' not in src and "'softanchor'" not in src:
    src = src.replace(
        "    if method in (\"lpa_protonet\", \"scp_lpa\"):\n"
        "        return LinguisticProtoNet()\n",
        "    if method in (\"lpa_protonet\", \"scp_lpa\"):\n"
        "        return LinguisticProtoNet()\n"
        "    if method in (\"softanchor\", \"sa_proto\"):\n"
        "        return SoftAnchorProtoNet()\n",
    )
    write(RUNNER, src)
    print("Added softanchor handler")
else:
    print("softanchor already in runner")