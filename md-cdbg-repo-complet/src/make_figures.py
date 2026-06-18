"""
make_figures.py — Regenerate all result figures from saved JSON results.

Reads ../results/scenario_{O,P,E}.json and produces the figures in
../figures/. Run run_experiments.py first to generate the JSON files.

Usage:
    python make_figures.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "../results"
FIG_DIR = "../figures"
SC_TAG = ["O", "P", "E"]
SC_NAMES = ["O (Opportunistic)", "P (Persistent)", "E (State-sponsored)"]
STRATS = ["No-Deception", "Random", "Asghar-2025", "R-BNE-DO+LCP"]
COLORS = {"No-Deception": "#95A5A6", "Random": "#F39C12",
          "Asghar-2025": "#27AE60", "R-BNE-DO+LCP": "#1D3461"}
LS = {"No-Deception": ":", "Random": "--", "Asghar-2025": "-.", "R-BNE-DO+LCP": "-"}
LW = {"No-Deception": 1.5, "Random": 1.5, "Asghar-2025": 2.0, "R-BNE-DO+LCP": 2.8}


def ci95(d):
    n = len(d)
    if n < 2:
        return np.mean(d), 0.0
    return np.mean(d), np.std(d, ddof=1) / np.sqrt(n) * 1.96


def load():
    allr = {}
    for t, tag in enumerate(SC_TAG):
        with open(os.path.join(RESULTS_DIR, f"scenario_{tag}.json")) as f:
            allr[t] = json.load(f)
    return allr


def fig_comparison(allr):
    x = np.arange(3); w = 0.20
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, metric, title, flip in [
        (axes[0], "UD", "(a) Defender utility $U^D$", False),
        (axes[1], "UA", "(b) $|U^A|$ (lower=better)", True),
        (axes[2], "VC", "(c) Violation rate VC (lower=better)", False)]:
        for si, s in enumerate(STRATS):
            means, errs = [], []
            for t in range(3):
                vals = np.abs(allr[t][s][metric]) if flip else allr[t][s][metric]
                m, e = ci95(vals); means.append(m); errs.append(e)
            bars = ax.bar(x + (si - 1.5) * w, means, w, yerr=errs,
                          color=COLORS[s], alpha=0.88, label=s,
                          error_kw={"linewidth": 1.0, "capsize": 2})
            if s == "R-BNE-DO+LCP":
                for b in bars:
                    b.set_edgecolor("#E55B13"); b.set_linewidth(2.0)
        ax.set_xticks(x); ax.set_xticklabels(["Sc O", "Sc P", "Sc E"])
        ax.set_title(title, fontsize=11); ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Four-method comparison (95% CI)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig_resilience(allr):
    T = len(allr[0]["R-BNE-DO+LCP"]["omega"][0])
    t_ax = np.arange(T)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for col in range(3):
        ax = axes[col]
        for s in STRATS:
            mu = np.mean(allr[col][s]["omega"], axis=0)
            ax.plot(t_ax, mu, lw=LW[s], color=COLORS[s], ls=LS[s], label=s)
        ax.axhline(0.55, color="black", lw=1.5, ls="--", alpha=0.7)
        ax.set_title(SC_NAMES[col], fontsize=10, fontweight="bold")
        ax.set_xlabel("Time step $t$"); ax.set_ylim(0.0, 1.05)
        if col == 0:
            ax.set_ylabel("$\\Omega(t)$", fontsize=11)
        ax.grid(True, alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=8.5)
    fig.suptitle("Global resilience $\\Omega(t)$ over time", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "resilience.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    allr = load()
    fig_comparison(allr)
    fig_resilience(allr)
    print(f"Figures saved to {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    main()
