"""
run_experiments.py — Reproduce the MD-CDBG experimental results.

Runs all strategies across the three attacker scenarios (O, P, E),
prints a summary table, and saves per-scenario results as JSON.

Usage:
    python run_experiments.py [--reps N] [--out DIR]

Example:
    python run_experiments.py --reps 10 --out ../results
"""
import argparse
import json
import os
import time
import numpy as np

from simulation import STRATEGIES

SC_NAMES = ["O (Opportunistic)", "P (Persistent)", "E (State-sponsored)"]
SC_TAG = ["O", "P", "E"]


def ci95(data):
    """Return mean and 95% confidence half-width."""
    n = len(data)
    m = float(np.mean(data))
    if n < 2:
        return m, 0.0
    se = float(np.std(data, ddof=1) / np.sqrt(n))
    return m, 1.96 * se


def main():
    parser = argparse.ArgumentParser(description="Run MD-CDBG experiments.")
    parser.add_argument("--reps", type=int, default=10,
                        help="Number of independent replications per scenario.")
    parser.add_argument("--out", type=str, default="../results",
                        help="Output directory for JSON results.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_results = {}

    t_start = time.time()
    for theta in range(3):
        all_results[theta] = {}
        print(f"\n=== Scenario {SC_NAMES[theta]} ({args.reps} reps) ===")
        for name, runner in STRATEGIES:
            reps = [runner(theta, seed=theta * 100 + r) for r in range(args.reps)]
            rec = {
                "UD": [r["UD"] for r in reps],
                "UA": [r["UA"] for r in reps],
                "VC": [r["VC"] for r in reps],
                "omega": [r["omega"] for r in reps],
                "belief": [r["belief"] for r in reps],
            }
            all_results[theta][name] = rec
            ud, ud_ci = ci95(rec["UD"])
            ua = float(np.mean(rec["UA"]))
            vc = float(np.mean(rec["VC"]))
            print(f"  {name:<16}: U^D={ud:6.3f}+/-{ud_ci:.3f}  "
                  f"U^A={ua:7.3f}  VC={vc:.3f}")

        # Save per-scenario JSON
        out_path = os.path.join(args.out, f"scenario_{SC_TAG[theta]}.json")
        with open(out_path, "w") as f:
            json.dump(all_results[theta], f)

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Method':<16} {'Sc':>3} {'U^D':>14} {'U^A':>8} {'VC':>7}")
    print("=" * 60)
    for theta in range(3):
        vc_asghar = np.mean(all_results[theta]["Asghar-2025"]["VC"])
        vc_rbne = np.mean(all_results[theta]["R-BNE-DO+LCP"]["VC"])
        for name, _ in STRATEGIES:
            d = all_results[theta][name]
            m, e = ci95(d["UD"])
            note = ""
            if name == "R-BNE-DO+LCP" and vc_asghar > 0:
                note = f"(-{100 * (vc_asghar - vc_rbne) / vc_asghar:.0f}% VC vs Asghar)"
            print(f"{name:<16} {SC_TAG[theta]:>3} {m:8.3f}+/-{e:.3f} "
                  f"{np.mean(d['UA']):8.3f} {np.mean(d['VC']):7.3f}  {note}")
        print()

    print(f"Total runtime: {(time.time() - t_start) / 60:.1f} min")
    print(f"Results saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
