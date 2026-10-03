"""Multi-N-way experiment for ADIBench.

Runs ProtoNet (best baseline) on all 3 datasets at 3-way and (where supported)
10-way, plus 5-way (already done). Reports granularity curve.
"""
import sys, os, json, time, subprocess, argparse
sys.path.insert(0, r"D:/dacd2026/adibench_v1")

# Use the existing runner
SCRIPT = r"C:/Users/zsndz/Desktop/dacd_bench_v4_20260826_153119/experiments/run_adibench_baseline.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["protonet"])
    ap.add_argument("--datasets", nargs="+", default=["nadi_18", "nadi_5", "amgadhasan_5"])
    ap.add_argument("--n_ways", nargs="+", type=int, default=[3, 10])
    ap.add_argument("--k_shots", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--n_episodes", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=200)
    args = ap.parse_args()

    cmd = [
        "python", "-u", SCRIPT,
        "--methods"] + args.methods + [
        "--datasets"] + args.datasets + [
        "--n_ways"] + [str(n) for n in args.n_ways] + [
        "--k_shots"] + [str(k) for k in args.k_shots] + [
        "--seeds"] + [str(s) for s in args.seeds] + [
        "--n_episodes", str(args.n_episodes),
        "--n_eval", str(args.n_eval),
    ]
    print("CMD:", " ".join(cmd))
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()