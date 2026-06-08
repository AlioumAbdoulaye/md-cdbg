# MD-CDBG: A Multi-Domain Cyber-Deception Bayesian Game for Resilient IoT Systems

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Reference implementation of the **Multi-Domain Cyber-Deception Bayesian Game
(MD-CDBG)** and the **R-BNE-DO** algorithm, accompanying the paper:

> A. Abdoulaye, A. M. Gueroui, A. A. Abba Ari, and Y. Benallouche,
> *"A Multi-Domain Cyber-Deception Bayesian Game for Resilient IoT Systems."*

---

## Overview

Advanced Persistent Threats (APTs) progress methodically across the three IoT
layers — perception, network, and application. Existing game-theoretic
deception models treat each layer in isolation, assume a zero-sum interaction,
and ignore resilience. The **MD-CDBG** addresses all three gaps:

- **Multi-domain coupling** of the three IoT layers via APT progression,
  semantic coherence, and a global budget constraint.
- **Non-zero-sum** modeling solved exactly with a **Linear Complementarity
  Program (LCP)** — not a zero-sum proxy.
- **Resilience as a hard constraint** through a new solution concept, the
  **Resilience-constrained Bayesian Nash Equilibrium (R-BNE)**.

The **R-BNE-DO** algorithm computes the equilibrium with a Bayesian Double
Oracle whose inner oracle uses the LCP, coordinated by a Lagrangian
decomposition that solves the three layers in parallel.

---

## Key result

R-BNE-DO+LCP reduces resilience-constraint violations by **57–69%** relative to
the state-of-the-art GameSec 2025 double-oracle approach, while inducing
**rational attacker behavior** (10 runs, 95% CI):

| Scenario | Method        | U^D   | U^A   | VC (↓ better) |
|----------|---------------|-------|-------|---------------|
| O        | Asghar-2025   | 5.83  | −1.93 | 0.45          |
| O        | **R-BNE-DO+LCP** | 5.62  | −0.68 | **0.18**   |
| P        | Asghar-2025   | 5.45  | −2.11 | 0.48          |
| P        | **R-BNE-DO+LCP** | 5.56  | −0.63 | **0.21**   |
| E        | Asghar-2025   | 5.33  | −2.11 | 0.58          |
| E        | **R-BNE-DO+LCP** | 5.79  | −0.56 | **0.18**   |

*VC = fraction of time steps with resilience below threshold. A less negative
U^A indicates a more rational (less self-defeating) attacker model.*

---

## Repository structure

```
md-cdbg/
├── src/
│   ├── simulation.py        # MD-CDBG model + R-BNE-DO (Algorithms 1 & 2) + baselines
│   ├── lcp_solver.py        # LCP solver (non-zero-sum Nash via support enumeration)
│   ├── run_experiments.py   # Run all scenarios, print table, save JSON
│   └── make_figures.py      # Regenerate figures from saved results
├── notebooks/
│   └── MD_CDBG_Simulation.ipynb   # Self-contained Google Colab notebook
├── figures/                 # Architecture, interaction, and result figures
├── results/                 # Saved JSON results per scenario
├── paper/                   # LaTeX source + PDF preview
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/AlioumAbdoulaye/md-cdbg.git
cd md-cdbg
pip install -r requirements.txt
```

Requires Python 3.8+ with NumPy, SciPy, and Matplotlib. No GPU needed.

---

## Usage

**Run all experiments** (3 scenarios × 4 methods × 10 reps):

```bash
cd src
python run_experiments.py --reps 10 --out ../results
```

**Regenerate the figures** from saved results:

```bash
python make_figures.py
```

**Run on Google Colab:** open `notebooks/MD_CDBG_Simulation.ipynb` and
*Run all*. Everything is self-contained (~10–20 min on a CPU runtime).

---

## Methods compared

| Method          | Description |
|-----------------|-------------|
| No-Deception    | Passive monitoring, no decoys (lower bound) |
| Random          | Uniform random decoy deployment |
| Asghar-2025     | GameSec 2025 double oracle (greedy + zero-sum LP) |
| **R-BNE-DO+LCP** | **This work** — Bayesian DO + LCP + resilience constraint |

---

## Citation

```bibtex
@inproceedings{abdoulaye2026mdcdbg,
  title     = {A Multi-Domain Cyber-Deception Bayesian Game for Resilient IoT Systems},
  author    = {Abdoulaye, Alioum and Gueroui, Abdelhak Mourad and
               Abba Ari, Ado Adamou and Benallouche, Yacine},
  booktitle = {Proc. ACM Int. Conf. on Modeling, Analysis and Simulation
               of Wireless and Mobile Systems (MSWiM), PhD Forum},
  year      = {2026}
}
```

---

## Acknowledgment

This work was supported by the French Government Scholarship (BGF) and conducted
within the cotutelle program between Université Paris-Saclay (Laboratoire DAVID)
and Université de Ngaoundéré.

## License

Released under the [MIT License](LICENSE).
