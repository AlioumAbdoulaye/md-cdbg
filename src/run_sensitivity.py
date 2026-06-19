#!/usr/bin/env python3
"""
run_sensitivity.py
==================
Reproduces the sensitivity analysis reported in the paper
(Section "Sensitivity Analysis").

It sweeps the two critical parameters of the MD-CDBG and records, for each
value, the mean resilience-constraint violation rate (VC) of every method:

    * Omega_min (resilience threshold)  in [0.45, 0.50, 0.55, 0.60, 0.65]
    * Budget B  (global decoy budget)   in [0.80, 1.00, 1.20, 1.40, 1.60]

For each parameter value the VC is averaged over the three attacker scenarios
(opportunistic / persistent / state-sponsored) and over several random seeds,
using the same article seeding convention  seed = theta*100 + r.

Outputs (written to ../results/):
    sens_omega.json   : VC vs Omega_min for all four methods
    sens_budget.json  : VC vs Budget B   for all four methods

These JSON files feed the two-panel sensitivity figure
(figures/sensitivity.png), produced by make_figures.py.

Usage:
    python run_sensitivity.py            # default: 5 reps
    python run_sensitivity.py --reps 10  # paper-grade (slower)

Note: the sweep temporarily overrides the module-level constants
OMEGA_MIN / BUDGET_B at run time; it always restores their defaults afterwards.
The source code of simulation.py itself is never modified.
"""

import argparse
import json
import os

import numpy as np

import simulation as sim

# Methods compared (key -> simulation entry point)
METHODS = {
    "rbne":   sim.run_rbne_do_lcp,    # proposed: R-BNE-DO + LCP
    "asghar": sim.run_asghar2025,     # strong baseline (GameSec 2025)
    "random": sim.run_random,         # weak baseline: random placement
    "nodec":  sim.run_no_deception,   # weak baseline: no deception
}

THETAS = [0, 1, 2]  # opportunistic, persistent, state-sponsored

# Parameter grids used in the paper
OMEGA_GRID = [0.45, 0.50, 0.55, 0.60, 0.65]
BUDGET_GRID = [0.80, 1.00, 1.20, 1.40, 1.60]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def mean_vc(method_fn, reps):
    """Mean VC over the 3 scenarios and `reps` seeds (article seeding)."""
    vcs = [
        method_fn(theta=theta, seed=theta * 100 + r)["VC"]
        for theta in THETAS
        for r in range(reps)
    ]
    return float(np.mean(vcs))


def sweep(param_name, grid, reps):
    """Sweep one parameter; return {param, rbne, asghar, random, nodec}."""
    out = {param_name: [], "rbne": [], "asghar": [], "random": [], "nodec": []}
    default = getattr(sim, "OMEGA_MIN" if param_name == "omega" else "BUDGET_B")
    attr = "OMEGA_MIN" if param_name == "omega" else "BUDGET_B"
    try:
        for value in grid:
            setattr(sim, attr, value)
            out[param_name].append(value)
            for key, fn in METHODS.items():
                out[key].append(mean_vc(fn, reps))
            red = 100 * (out["asghar"][-1] - out["rbne"][-1]) / out["asghar"][-1]
            print(f"  {attr}={value:<5} VC: R-BNE={out['rbne'][-1]:.3f} "
                  f"Asghar={out['asghar'][-1]:.3f} (reduction {red:.0f}%)")
    finally:
        setattr(sim, attr, default)  # always restore the default
    return out


def main():
    parser = argparse.ArgumentParser(description="MD-CDBG sensitivity analysis")
    parser.add_argument("--reps", type=int, default=5,
                        help="random seeds per scenario (paper uses 5-10)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Sensitivity to Omega_min  (reps={args.reps}):")
    res_omega = sweep("omega", OMEGA_GRID, args.reps)
    with open(os.path.join(RESULTS_DIR, "sens_omega.json"), "w") as f:
        json.dump(res_omega, f, indent=2)

    print(f"\nSensitivity to Budget B  (reps={args.reps}):")
    res_budget = sweep("budget", BUDGET_GRID, args.reps)
    with open(os.path.join(RESULTS_DIR, "sens_budget.json"), "w") as f:
        json.dump(res_budget, f, indent=2)

    print("\nDone. Wrote sens_omega.json and sens_budget.json to results/.")


if __name__ == "__main__":
    main()
