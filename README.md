# MD-CDBG: Multi-Domain Cyber-Deception Bayesian Game for Resilient IoT Systems

This repository contains the simulation code, experimental results, and figures
for the paper:

> **A Multi-Domain Cyber-Deception Bayesian Game for Resilient IoT Systems**
> Alioum Abdoulaye, Zainab Issa Ahmad, Yacine Benallouche, 
> Abdelhak Mourad Gueroui, Ado Adamou Abba Ari.

## Overview

The **MD-CDBG** models cyber-deception across the three layers of an IoT system
(perception, network, application) as three coupled Bayesian sub-games. The
defender deploys decoys (honeypots) to mislead an attacker whose type is unknown,
while a **hard resilience constraint** must hold at every time step.

The solution concept is the **Resilience-constrained Bayesian Nash Equilibrium
(R-BNE)**, solved by a Bayesian Double Oracle combined with a Linear
Complementarity Program (LCP) that handles the non-zero-sum structure of the game.

Three coupling constraints link the layers:
- **C1** — APT progression (perception -> network -> application);
- **C2** — semantic coherence between adjacent-layer decoys;
- **C3** — a global deployment budget.

## Repository structure

```
.
├── src/
│   ├── simulation.py        # MD-CDBG model, R-BNE-DO+LCP, and all baselines
│   ├── lcp_solver.py        # LCP solver (support enumeration)
│   ├── run_experiments.py   # main comparison (Table III scenarios O/P/E)
│   ├── run_sensitivity.py   # sensitivity sweeps over Omega_min and budget B
│   └── make_figures.py      # regenerates the result figures
├── results/                 # JSON outputs (scenarios + sensitivity sweeps)
├── figures/                 # figures used in the paper
├── notebooks/
│   └── MD_CDBG_Simulation.ipynb   # self-contained Colab notebook
└── Papers/                  # manuscript (LaTeX), bibliography, project PDF
```

## Methods compared

| Method | Description |
|--------|-------------|
| **R-BNE-DO+LCP** | Proposed: resilience-constrained equilibrium via Double Oracle + LCP |
| Asghar-2025 | State-of-the-art double-oracle deception baseline |
| Random | Random decoy placement |
| No-Deception | No decoys deployed |

## Reproducing the results

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main comparison (scenarios O, P, E):

```bash
cd src
python run_experiments.py
```

Run the sensitivity analysis (sweeps Omega_min and budget B):

```bash
cd src
python run_sensitivity.py            # 5 seeds per point
python run_sensitivity.py --reps 10  # paper-grade (slower)
```

Regenerate the figures:

```bash
cd src
python make_figures.py
```

Or open the self-contained notebook `notebooks/MD_CDBG_Simulation.ipynb`
in Google Colab to reproduce the main results end-to-end.

## Key results

- R-BNE-DO+LCP reduces resilience-constraint violations by **65–72%** relative
  to the strongest baseline across three attacker profiles.
- The sensitivity analysis confirms this advantage is **robust**: the reduction
  stays in the **70–77%** band as `Omega_min` varies over [0.45, 0.65], and in
  the **71–77%** band as the budget `B` varies over [0.8, 1.6]. The ordering
  R-BNE < Random < Asghar < No-Deception never changes.

## License

See the `LICENSE` file.
