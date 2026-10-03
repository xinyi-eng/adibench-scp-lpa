"""Run RelationNet multi-seed sweep."""
import sys, os, json, subprocess, time

RUNNER = r"D:/dacd2026/adibench_v1/experiments/run_adibench_baseline.py"

combos = []
for ds in ["nadi_18", "nadi_5", "amgadhasan_5"]:
    for k in [1, 5]:
        for s in [0, 7, 42, 123]:
            combos.append((ds, k, s))

total = len(combos)
done = 0
t0 = time.time()

for ds, k, s in combos:
    out = f"D:/dacd2026/adibench_v1/results/relationnet_{ds}_5way_{k}shot_seed{s}.json"
    if os.path.exists(out):
        with open(out) as f:
            d = json.load(f)
        if "accuracy" in d:
            done += 1
            print(f"[skip {done}/{total}] {os.path.basename(out)} (existing {d['accuracy']:.4f})", flush=True)
            continue
    cmd = [sys.executable, RUNNER,
           "--methods", "relationnet",
           "--datasets", ds,
           "--n_ways", "5",
           "--k_shots", str(k),
           "--seeds", str(s),
           "--n_episodes", "500",
           "--n_eval", "200"]
    print(f"[{done+1}/{total}] running {ds} {k}shot seed{s}...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED rc={r.returncode}; stderr: {r.stderr[-200:]}", flush=True)
    else:
        tail = [l for l in r.stdout.split("\n") if l.strip()][-1:]
        print(f"  OK | {tail[0][:120] if tail else ''}", flush=True)
    done += 1

elapsed = time.time() - t0
print(f"\n=== RelationNet multi-seed DONE: {done}/{total} in {elapsed/60:.1f} min ===")
