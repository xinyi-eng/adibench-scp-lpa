"""Verify and fix runner import for SoftAnchor."""
import io

RUNNER = r"D:/dacd2026/adibench_v1/experiments/run_adibench_baseline.py"


def read(p):
    with io.open(p, "r", encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


src = read(RUNNER)
if "SoftAnchorProtoNet" not in src:
    # Add to import
    src = src.replace(
        "from adibench.baselines import (\n"
        "    ProtoNet, MatchingNet, RelationNet, FineTune, Focal, CB, MAML, Random,\n"
        "    CFProtoNet, cf_protonet_supcon_step, LinguisticProtoNet, lpa_protonet_step,\n"
        ")",
        "from adibench.baselines import (\n"
        "    ProtoNet, MatchingNet, RelationNet, FineTune, Focal, CB, MAML, Random,\n"
        "    CFProtoNet, cf_protonet_supcon_step, LinguisticProtoNet, lpa_protonet_step,\n"
        "    SoftAnchorProtoNet, softanchor_step,\n"
        ")",
    )
    write(RUNNER, src)
    print("Added import")
else:
    print("Import already present")