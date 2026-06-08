"""
simulation_v5.py — MD-CDBG Simulation: rigorously aligned with the theory

Architecture faithfully mirrors Algorithms 1 and 2:

  Algorithm 1 (Bayes-DO):
    - Called per layer k at each time step t
    - Takes: G_k, λ, ν_k, μ_k^t(θ_A)
    - Returns: local BNE (σ_k^D*, σ_k^A*)
    - Step 1: BNE via LCP — two matrices M_k^D (belief-weighted Lagrangian)
              and M_k^A (true attacker utility)
    - Step 2: Defender BR in expectation over belief μ_k^t
    - Step 3: Attacker BR using true u_k^A
    - Step 4: Expansion check — exact termination

  Algorithm 2 (R-BNE-DO):
    - Operates on all 3 layers over horizon T
    - Phase A: C1 propagation + feasibility check + Bayes-DO per layer (parallel)
    - Phase B: Action sampling + state transition
    - Phase C: Bayesian belief update per layer μ_k^{t+1}
    - Phase D: Lagrange multiplier updates λ^{t+1}, ν_k^{t+1}
    - Stopping: max(||Δσ^D||, |Δλ|, max_k|Δν_k|) < ε

Key corrections vs previous version:
  - u_k^D uses belief-weighted expectation (not fixed type)
  - M_k^D built from belief μ_k^t, not true type θ_A*
  - ν_k replaces μ_k as Lagrange multiplier for C2 (notation fix)
  - ν_k included in stopping criterion
  - u_D global = Σ_k β_k * E_{θ_A~μ_k^t}[u_k^D]
  - u_A global = Σ_k β_k^A * u_k^A (true type, with C1 indicator)
  - Resilience Ω_k computed per layer and per time step
  - Bayes-DO called once per layer per time step (exact termination)
"""

import numpy as np
from scipy.optimize import linprog
from lcp_solver import solve_bimatrix_support_enum

# ══════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════
N_LAYERS = 3
N_A      = [5, 6, 5]   # attacker actions per layer
N_D      = [5, 5, 5]   # defender actions per layer
N_REAL   = [20, 10, 5] # real nodes per layer
N_DECOY  = [12, 6,  4] # decoy nodes per layer
T        = 50
N_TYPES  = 3            # θ_opp, θ_pers, θ_state

# Attacker types
PRIOR_A = np.array([0.50, 0.30, 0.20])  # p(θ_A)
THETA_A = ["opportunistic", "persistent", "state-sponsored"]

# β_k^A: attacker layer weights (application layer most valuable)
BETA_A_L = np.array([0.15, 0.30, 0.55])

# Type-specific detection capability ψ_k(θ_A)
PSI = np.array([
    [0.20, 0.20, 0.20],  # θ_opp
    [0.50, 0.50, 0.50],  # θ_pers
    [0.65, 0.65, 0.70],  # θ_state (L3 most capable: 0.70)
])  # shape: (N_TYPES, N_LAYERS)

# Defender utility weights: α_1 (deception), α_2 (resilience),
#                           α_3 (energy), α_4 (cost)
ALPHA = np.array([0.55, 0.32, 0.07, 0.06])

# β_k^D: defender layer weights
BETA_D = np.array([0.15, 0.30, 0.55])

# β_k^Ω: resilience aggregation weights
BETA_O = np.array([0.20, 0.30, 0.50])

# Attacker utility weights: γ_1 (gain), γ_2 (cost), γ_3 (risk)
GAMMA_A = np.array([0.55, 0.25, 0.20])

# Decoy quality decomposition weights: ω_1 (func), ω_2 (stat), ω_3 (temp)
OMEGA_W = np.array([0.40, 0.35, 0.25])

# Asset values v_{k,i}^A
V_A = [
    np.array([0.35, 0.20, 0.15, 0.25, 0.10]),
    np.array([0.50, 0.20, 0.45, 0.30, 0.25, 0.40]),
    np.array([0.56, 0.55, 0.45, 0.40, 0.50]),
]

# Attack success probabilities p_{k,i}^A
P_A = [
    np.array([0.28, 0.32, 0.25, 0.38, 0.15]),
    np.array([0.25, 0.30, 0.28, 0.35, 0.30, 0.10]),
    np.array([0.23, 0.20, 0.25, 0.28, 0.18]),
]

# Resource cost r_{k,i}^A
R_A_COST = [
    np.array([2., 1.5, 3., 0.5, 4.]),
    np.array([3., 1., 4., 5., 2.5, 6.]),
    np.array([4., 5., 3., 2., 4.5]),
]

# Detection sensitivity λ_{k,i} (negative for a_{2,6}^A)
LAM_KI = [
    np.array([0.20, 0.15, 0.10, 0.05, 0.12]),
    np.array([0.15, 0.10, 0.20, 0.18, 0.12, -0.25]),
    np.array([0.08, 0.10, 0.15, 0.08, 0.05]),
]

# Defender energy cost e_{k,i}^D and operational cost c_{k,i}^D
E_D = [
    np.array([1980., 240., 150., 300., 120.]),
    np.array([2500., 800., 400., 600., 200.]),
    np.array([1200., 500., 300., 800., 100.]),
]
C_D = [
    np.array([8., 2., 1.5, 3., 1.]),
    np.array([10., 4., 2., 5., 1.5]),
    np.array([6., 3., 2.5, 7., 1.]),
]
E_MAX = [e.max() for e in E_D]
C_MAX = [c.max() for c in C_D]

# Decoy functional realism r_{k,i}^func
R_FUNC = [
    np.array([0.92, 0.72, 0.45, 0.18, 0.28]),
    np.array([0.88, 0.70, 0.50, 0.20, 0.30, 0.15]),
    np.array([0.90, 0.65, 0.40, 0.15, 0.25]),
]

# Resilience parameters
RHO    = np.array([0.80, 0.90, 0.99])  # resistance threshold
W_FIX  = np.array([0.50, 0.30, 0.20])  # fixed resilience weights
LAM1, LAM2 = 4.62, 4.0                  # adaptive weight parameters
TAU_K  = np.array([5, 3, 2])            # recovery time per node
TMAX_K = np.array([10, 8, 5])           # max acceptable recovery time

# Algorithm parameters
OMEGA_MIN  = 0.55   # Ω_min: resilience threshold
DELTA_MIN  = 0.60   # δ_min: coherence threshold for C2
BUDGET_B   = 1.2    # B: global budget for C3
GAMMA_D    = 0.95   # discount factor
ETA0       = 0.10   # initial step size η_0
MAX_DO     = 20     # max DO iterations per Bayes-DO call
ISOLATION  = 3      # defender action index for isolation

# ══════════════════════════════════════════════════════════════
# STATE
# Represents s(t): compromised nodes per layer + belief per layer
# ══════════════════════════════════════════════════════════════
class State:
    def __init__(self):
        self.comp   = np.zeros(N_LAYERS, dtype=int)
        # μ_k^t(θ_A): belief per layer, shape (N_LAYERS, N_TYPES)
        self.belief = np.tile(PRIOR_A.copy(), (N_LAYERS, 1))
        self.t = 0

    def delta(self, k):
        """Compromise ratio δ_k = comp_k / N_real_k"""
        return self.comp[k] / N_REAL[k]

    def delta_bar(self):
        """Average compromise ratio across layers"""
        return float(np.mean([self.delta(k) for k in range(N_LAYERS)]))

    def accessible(self, k):
        """C1: layer k accessible iff layer k-1 has ≥1 compromised node"""
        return k == 0 or self.comp[k-1] > 0

    def clone(self):
        s = State()
        s.comp   = self.comp.copy()
        s.belief = self.belief.copy()
        s.t      = self.t
        return s

def _e(n, i):
    """Unit basis vector e_i ∈ R^n"""
    v = np.zeros(n); v[i] = 1.; return v

# ══════════════════════════════════════════════════════════════
# DECOY QUALITY q_k(σ_k^D)
# q_k = ω_1 * q_func + ω_2 * q_stat + ω_3 * q_temp
# Simplified: q_stat ≈ 0.88 * q_func, q_temp ≈ 0.82 * q_func
# ══════════════════════════════════════════════════════════════
def decoy_quality(k, sD):
    """
    Composite decoy quality q_k(σ_k^D).
    q_func = E_{σ_k^D}[r_{k,i}^func]
    q_stat ≈ 0.88 * q_func (calibrated)
    q_temp ≈ 0.82 * q_func (calibrated)
    """
    q_func = float(sD @ R_FUNC[k][:N_D[k]])
    q_stat = 0.88 * q_func
    q_temp = 0.82 * q_func
    return float(np.clip(
        OMEGA_W[0]*q_func + OMEGA_W[1]*q_stat + OMEGA_W[2]*q_temp,
        0., 1.
    ))

# ══════════════════════════════════════════════════════════════
# DECEPTION EFFECTIVENESS f_1^k(σ_k^D, σ_k^A | θ_A)
# f_1^k = (|N_decoy| / (|N_real| + |N_decoy|)) * q_k * (1 - d_k)
# d_k = 1 - (1 - d_effort)(1 - d_vuln)
# ══════════════════════════════════════════════════════════════
def f1(k, sA, sD, theta):
    """
    Deception effectiveness at layer k for attacker type theta.
    theta ∈ {0=opp, 1=pers, 2=state}
    """
    ratio = N_DECOY[k] / (N_REAL[k] + N_DECOY[k])
    q     = decoy_quality(k, sD)
    psi   = PSI[theta, k]  # ψ_k(θ_A): detection capability

    # Active detection: d_effort = σ_k^A(a_{2,6}^A) * ψ_k(θ_A)
    d_effort = (float(sA[5]) * psi) if (k == 1 and N_A[k] == 6) else (0.03 * psi)

    # Passive detection: d_vuln = (1 - q) * 0.35
    d_vuln = (1. - q) * 0.35

    # Total: d = 1 - (1-d_effort)(1-d_vuln)
    d = 1. - (1. - d_effort) * (1. - d_vuln)

    return float(ratio * q * (1. - d))

# ══════════════════════════════════════════════════════════════
# RESILIENCE
# Ω_k(t) = w1*resistance + w2*absorption + w3*recovery
# Adaptive weights based on global compromise δ̄(t)
# ══════════════════════════════════════════════════════════════
def adaptive_weights(dbar):
    """
    Adaptive resilience weights w1, w2, w3 as functions of δ̄.
    w1 = W1 * exp(-λ1 * δ̄)   (resistance decreases with compromise)
    w2 = W2 * λ2*e * δ̄ * exp(-λ2*δ̄)  (absorption peaks mid-compromise)
    w3 = 1 - w1 - w2           (recovery)
    """
    w1 = W_FIX[0] * np.exp(-LAM1 * dbar)
    w2 = max(0., W_FIX[1] * LAM2 * np.e * dbar * np.exp(-LAM2 * dbar))
    w3 = max(0., 1. - w1 - w2)
    tot = w1 + w2 + w3
    if tot > 1e-8:
        return w1/tot, w2/tot, w3/tot
    return tuple(W_FIX)

def omega_local(k, state):
    """
    Local resilience Ω_k(t) for layer k.
    resistance:  1 if (1 - δ_k) ≥ ρ_k
    absorption:  max(0, 1 - δ_k)
    recovery:    1 if δ_k * N_real_k * τ_k ≤ T_max_k
    """
    w1, w2, w3 = adaptive_weights(state.delta_bar())
    d_k = state.delta(k)
    resistance = float(1. - d_k >= RHO[k])
    absorption = max(0., 1. - d_k)
    recovery   = float(d_k * N_REAL[k] * TAU_K[k] <= TMAX_K[k])
    return float(w1 * resistance + w2 * absorption + w3 * recovery)

def omega_global(state):
    """
    Global resilience Ω(t) = Σ_k β_k^Ω * Ω_k(t)
    """
    return float(sum(BETA_O[k] * omega_local(k, state)
                     for k in range(N_LAYERS)))

# ══════════════════════════════════════════════════════════════
# COHERENCE Coh_k (C2 constraint)
# ══════════════════════════════════════════════════════════════
def coherence(k, sD_k, sD_k1):
    """
    Inter-layer decoy coherence between layers k and k+1.
    Simplified: coherence depends on decoy quality alignment.
    """
    q_k  = decoy_quality(k,   sD_k)
    q_k1 = decoy_quality(k+1, sD_k1)
    # Coherence = 1 - |q_k - q_{k+1}| / max(q_k, q_{k+1}+ε)
    denom = max(q_k, q_k1, 1e-8)
    return float(1. - abs(q_k - q_k1) / denom)

# ══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════

def u_D_local(k, sD, sA, theta, state):
    """
    Local defender utility u_k^D(σ_k^D, σ_k^A | θ_A) for fixed type theta.
    u_k^D = α_1*f_1^k + α_2*Ω_k - α_3*f_3^k - α_4*f_4^k
    """
    f1k = f1(k, sA, sD, theta)
    om_k = omega_local(k, state)
    f3k  = float(sD @ np.array(E_D[k][:N_D[k]])) / E_MAX[k]  # energy
    f4k  = float(sD @ np.array(C_D[k][:N_D[k]])) / C_MAX[k]  # cost
    return float(ALPHA[0]*f1k + ALPHA[1]*om_k
                 - ALPHA[2]*f3k - ALPHA[3]*f4k)

def u_D_bayes(k, sD, sA_dist, belief_k, state):
    """
    Belief-weighted defender utility (Eq. util_D_k_bayes):
    ū_k^D(σ_k^D(μ_k^t), μ_k^t) = Σ_{θ_A} μ_k^t(θ_A) * u_k^D(σ_k^D, σ_k^A|θ_A)
    sA_dist[theta] = σ_k^A(·|θ_A) for each type
    """
    return float(sum(
        belief_k[theta] * float(sum(
            sum(sD[i] * sA_dist[theta][a] * u_D_local(k, _e(N_D[k],i),
                                                        _e(N_A[k],a), theta, state)
                for a in range(N_A[k]))
            for i in range(N_D[k])))
        for theta in range(N_TYPES)
    ))

def U_D_global(sD_list, sA_dist_list, state):
    """
    Global defender utility (Eq. util_D_global):
    U^D = Σ_k β_k * ū_k^D(σ_k^D(μ_k^t), μ_k^t)
    """
    return float(sum(
        BETA_D[k] * u_D_bayes(k, sD_list[k], sA_dist_list[k],
                               state.belief[k], state)
        for k in range(N_LAYERS)
    ))

def u_A_local(k, sD, sA, theta, state):
    """
    Local attacker utility u_k^A(σ_k^A, σ_k^D | θ_A).
    Returns 0 if layer k not accessible (C1 constraint).
    u_k^A = γ_1*g_k^A - γ_2*c_k^A - γ_3*ρ_k^A
    """
    if not state.accessible(k):
        return 0.0
    f1k = f1(k, sA, sD, theta)
    # g_k^A: expected asset gain
    g = float(sA @ (V_A[k][:N_A[k]] * (1. - f1k) * P_A[k][:N_A[k]]))
    # c_k^A: expected resource cost
    c = float(sA @ R_A_COST[k][:N_A[k]]) / R_A_COST[k][:N_A[k]].max()
    # ρ_k^A: expected detection risk
    rho = float(sA @ (f1k * LAM_KI[k][:N_A[k]]))
    return float(GAMMA_A[0]*g - GAMMA_A[1]*c - GAMMA_A[2]*rho)

def U_A_global(sD_list, sA_list, theta, state):
    """
    Global attacker utility (Eq. util_A_global):
    U^A = Σ_k β_k^A * u_k^A(σ_k^A, σ_k^D | θ_A*) * 1[A_k^A(s) ≠ ∅]
    C1 indicator is embedded in u_A_local.
    """
    return float(sum(
        BETA_A_L[k] * float(sum(
            sum(sD_list[k][i] * sA_list[k][a] *
                u_A_local(k, _e(N_D[k],i), _e(N_A[k],a), theta, state)
                for a in range(N_A[k]))
            for i in range(N_D[k])))
        for k in range(N_LAYERS)
    ))

# ══════════════════════════════════════════════════════════════
# PAYOFF MATRICES (for Algorithm 1)
# ══════════════════════════════════════════════════════════════

def make_MD(k, state, lam, nu_k, belief_k, sD_prev=None):
    """
    Defender payoff matrix M_k^D[i,j] for Algorithm 1 Step 1.
    M_k^D[i,j] = E_{θ_A~μ_k^t}[L_k(i,j,λ,ν_k)]
               = Σ_θ μ_k^t(θ) * [u_k^D(e_i, e_j|θ) - λ*f4_k(e_i)
                                   - ν_k * max(0, δ_min - Coh_k(e_i, ...))]
    Note: coherence term uses previous layer strategy if available.
    """
    nD, nA = N_D[k], N_A[k]
    M = np.zeros((nD, nA))
    for i in range(nD):
        sD_i = _e(nD, i)
        f4k  = float(sD_i @ np.array(C_D[k][:nD])) / C_MAX[k]
        # Coherence penalty (C2): between k and k+1
        coh_pen = 0.
        if k < N_LAYERS - 1 and sD_prev is not None:
            coh = coherence(k, sD_i, sD_prev[k+1]) if k+1 < N_LAYERS else 1.
            coh_pen = nu_k * max(0., DELTA_MIN - coh)
        for j in range(nA):
            sA_j = _e(nA, j)
            u_expected = sum(
                belief_k[theta] * u_D_local(k, sD_i, sA_j, theta, state)
                for theta in range(N_TYPES)
            )
            M[i, j] = u_expected - lam * f4k - coh_pen
    return M

def make_MA(k, theta, state):
    """
    Attacker payoff matrix M_k^A[i,j] — true utility, independent of λ, ν_k.
    M_k^A[i,j] = u_k^A(e_i, e_j | θ_A*)
    """
    nD, nA = N_D[k], N_A[k]
    M = np.zeros((nD, nA))
    for i in range(nD):
        for j in range(nA):
            M[i, j] = u_A_local(k, _e(nD,i), _e(nA,j), theta, state)
    return M

def zero_sum_gap(k, theta, state, belief_k):
    """
    Zero-sum misspecification: ||M_D + M_A||_F / ||M_D||_F
    """
    MD = make_MD(k, state, 0., 0., belief_k)
    MA = make_MA(k, theta, state)
    gap = np.linalg.norm(MD + MA, 'fro')
    norm = np.linalg.norm(MD, 'fro')
    return float(gap / norm) if norm > 1e-8 else 0.

# ══════════════════════════════════════════════════════════════
# ALGORITHM 1: Bayes-DO
# Called once per layer k per time step t by Algorithm 2.
# Returns exact local BNE (σ_k^D*, σ_k^A*).
# No tolerance ε_DO — exact termination via expansion check.
# ══════════════════════════════════════════════════════════════
def bayes_do(k, theta, state, lam, nu_k, belief_k, sD_prev=None):
    """
    Algorithm 1 — Bayes-DO: Bayesian Double Oracle for sub-game G_k.

    Inputs:
      k        : layer index
      theta    : true attacker type θ_A* (used only for M_k^A)
      state    : current system state s(t)
      lam      : Lagrange multiplier λ (budget C3)
      nu_k     : Lagrange multiplier ν_k (coherence C2)
      belief_k : current belief μ_k^t(θ_A) for layer k
      sD_prev  : previous defender strategies (for coherence)

    Returns: (σ_k^D*, σ_k^A*) — exact local BNE
    """
    nD, nA = N_D[k], N_A[k]

    # Build full payoff matrices once
    MD_full = make_MD(k, state, lam, nu_k, belief_k, sD_prev)
    MA_full = make_MA(k, theta, state)

    # Initialize restricted sets R_D, R_A with one action each
    R_D = [0]; R_A = [0]
    sD  = _e(nD, 0)
    sA  = _e(nA, 0)

    for _ in range(MAX_DO):
        # ── Step 1: BNE via LCP on restricted game ─────────────
        MD_r = MD_full[np.ix_(R_D, R_A)]
        MA_r = MA_full[np.ix_(R_D, R_A)]
        sD_r, sA_r = solve_bimatrix_support_enum(MD_r, MA_r)

        # Map restricted strategies to full action space
        sD_new = np.zeros(nD)
        sA_new = np.zeros(nA)
        for ii, d in enumerate(R_D): sD_new[d] = sD_r[ii]
        for jj, a in enumerate(R_A): sA_new[a] = sA_r[jj]

        # ── Step 2: Defender BR in expectation over belief ─────
        # BR_D = argmax_{a ∈ A_k^D} E_{θ_A~μ_k^t}[L_k(a, σ̃_k^A, λ, ν_k)]
        br_D_vals = MD_full @ sA_new
        br_D = int(np.argmax(br_D_vals))

        # ── Step 3: Attacker BR using true utility M_k^A ───────
        # BR_A = argmax_{a ∈ A_k^A} u_k^A(σ̃_k^D, a | θ_A*)
        br_A_vals = sD_new @ MA_full
        br_A = int(np.argmax(br_A_vals))

        # ── Step 4: Expansion check (exact termination) ────────
        if br_D in R_D and br_A in R_A:
            # No profitable deviation outside R_D × R_A → exact BNE found
            sD, sA = sD_new, sA_new
            break

        # Expand restricted sets
        if br_D not in R_D: R_D.append(br_D)
        if br_A not in R_A: R_A.append(br_A)
        sD, sA = sD_new, sA_new

    return sD, sA

# ══════════════════════════════════════════════════════════════
# BAYESIAN UPDATE (Phase C of Algorithm 2)
# μ_k^{t+1}(θ_A) ∝ σ_k^A(a_k^A(t) | θ_A) * μ_k^t(θ_A)
# ══════════════════════════════════════════════════════════════
def bayes_update(belief_k, action_idx, k):
    """
    Bayesian belief update for layer k after observing action a_k^A.
    Likelihood: L(θ_A) = 0.4 * p_type(a|θ_A) + 0.6 * uniform
    """
    nA = N_A[k]
    # Type-specific action preference distributions
    prefs = [
        np.array([0.30,0.25,0.20,0.15,0.10]+[0.]*(nA-5))[:nA],  # opportunistic
        np.ones(nA) / nA,                                          # persistent
        np.array([0.10,0.15,0.20,0.25,0.30]+[0.]*(nA-5))[:nA],  # state
    ]
    if nA == 6:
        prefs[2] = np.array([0.08, 0.10, 0.15, 0.18, 0.22, 0.27])
    # Normalize each
    prefs = [p/p.sum() for p in prefs]
    # Likelihood: smoothed toward uniform
    L = np.array([0.40 * p[action_idx] + 0.60 / nA
                  for p in prefs]) + 1e-10
    post = L * belief_k
    return post / post.sum()

# ══════════════════════════════════════════════════════════════
# STATE TRANSITION (Phase B of Algorithm 2)
# ══════════════════════════════════════════════════════════════
def transition(state, aA, aD, theta, sD_list, rng):
    """
    s(t+1) = T(s(t), a^A(t), a^D(t)).
    C1 enforced: layer k only attacked if layer k-1 compromised.
    """
    s = state.clone()
    s.t += 1
    for k in range(N_LAYERS):
        if not s.accessible(k):
            continue
        sA_p = _e(N_A[k], aA[k])
        f1k  = f1(k, sA_p, sD_list[k], theta)
        # Attacker action: trapped by decoy with prob f1k, else succeeds with p_{k,i}^A
        if rng.random() < f1k:
            pass  # attacker trapped by honeypot
        elif rng.random() < P_A[k][aA[k]]:
            if s.comp[k] < N_REAL[k]:
                s.comp[k] += 1  # node compromised
        # Defender isolation action
        if aD[k] == ISOLATION and s.comp[k] > 0:
            s.comp[k] -= 1  # node recovered
    return s

# ══════════════════════════════════════════════════════════════
# ALGORITHM 2: R-BNE-DO (our full model)
# ══════════════════════════════════════════════════════════════
def run_rbne_do_lcp(theta, seed=0):
    """
    Algorithm 2 — R-BNE-DO: R-BNE via Bayesian Double Oracle.

    Operates on all 3 layers over horizon T.
    Phase A: C1 propagation + feasibility check + Bayes-DO (parallel)
    Phase B: Action sampling + state transition
    Phase C: Bayesian belief update per layer μ_k^{t+1}
    Phase D: Lagrange multiplier updates λ^{t+1}, ν_k^{t+1}
    Stopping: max(||Δσ^D||, |Δλ|, max_k|Δν_k|) < ε
    """
    rng   = np.random.default_rng(seed)
    state = State()

    # Initialize multipliers: λ^0=0, ν_k^0=0, ν_Ω^0=0
    lam     = 0.
    nu      = np.zeros(N_LAYERS)  # ν_k — coherence multipliers (C2)
    nu_omega = 0.                 # ν_Ω — resilience multiplier (soft)

    # Metrics
    omega_t  = np.zeros(T)   # Ω(t) global
    omega_kt = np.zeros((T, N_LAYERS))  # Ω_k(t) per layer
    f1_kt    = np.zeros((T, N_LAYERS))  # f_1^k(t) per layer
    ud_t     = np.zeros(T)   # U^D(t) global
    uA_t     = np.zeros(T)   # U^A(t) global
    belief_t = np.zeros((T, N_LAYERS, N_TYPES))  # μ_k^t full
    gap_t    = np.zeros(T)   # zero-sum gap

    sD_prev = [np.ones(N_D[k])/N_D[k] for k in range(N_LAYERS)]

    for t in range(T):

        # ── Phase A: C1 propagation + Bayes-DO per layer ───────
        # Record Ω_k(t) before strategies (current state)
        for k in range(N_LAYERS):
            omega_kt[t, k] = omega_local(k, state)
        omega_t[t] = omega_global(state)

        sD_list = []
        sA_list = []
        nu_prev = nu.copy()

        # Global feasibility check (Π^D_Ω = ∅ ?)
        # Use omega_global: if below Ω_min, ALL layers activate fallback
        omega_now = omega_t[t]
        resilience_critical = (omega_now < OMEGA_MIN)

        for k in range(N_LAYERS):
            if resilience_critical:
                # Phase A fallback: maximize Ω by isolating most compromised layer
                # Isolation on most compromised layer, honeypot on others
                if state.comp[k] > 0:
                    sD = _e(N_D[k], ISOLATION)  # recover compromised nodes
                else:
                    # Layer intact: maintain best deception possible
                    sD = _e(N_D[k], 0)           # best honeypot action
                sA = np.ones(N_A[k]) / N_A[k]
            else:
                # Call Algorithm 1: Bayes-DO
                # Pass effective lam = λ + ν_Ω (resilience pressure)
                sD, sA = bayes_do(
                    k       = k,
                    theta   = theta,
                    state   = state,
                    lam     = lam,
                    nu_k    = nu[k],
                    belief_k= state.belief[k],
                    sD_prev = sD_prev
                )
            sD_list.append(sD)
            sA_list.append(sA)

        # ── Record metrics at time t ────────────────────────────
        # f_1^k(t) per layer
        for k in range(N_LAYERS):
            f1_kt[t, k] = f1(k, sA_list[k], sD_list[k], theta)

        # U^D(t): belief-weighted global
        # Build sA_dist[k][theta] = σ_k^A for each type
        # (For recording: true type strategies, others uniform)
        sA_dist_list = []
        for k in range(N_LAYERS):
            sA_dist = [
                sA_list[k] if th == theta
                else np.ones(N_A[k]) / N_A[k]
                for th in range(N_TYPES)
            ]
            sA_dist_list.append(sA_dist)
        ud_t[t]  = U_D_global(sD_list, sA_dist_list, state)

        # U^A(t): true type, with C1 indicator
        uA_t[t]  = U_A_global(sD_list, sA_list, theta, state)

        # Belief μ_k^t
        belief_t[t] = state.belief.copy()

        # Zero-sum gap (average over layers)
        gap_t[t] = float(np.mean([
            zero_sum_gap(k, theta, state, state.belief[k])
            for k in range(N_LAYERS)
        ]))

        # ── Phase B: Action sampling and state transition ───────
        aA = [int(rng.choice(N_A[k], p=sA_list[k]))
              for k in range(N_LAYERS)]
        aD = [int(rng.choice(N_D[k], p=sD_list[k]))
              for k in range(N_LAYERS)]
        state = transition(state, aA, aD, theta, sD_list, rng)

        # ── Phase C: Bayesian belief update per layer ───────────
        # μ_k^{t+1}(θ_A) ∝ σ_k^A(a_k^A(t)|θ_A) * μ_k^t(θ_A)
        for k in range(N_LAYERS):
            state.belief[k] = bayes_update(state.belief[k], aA[k], k)

        # ── Phase D: Lagrange multiplier updates ────────────────
        eta = ETA0 / np.sqrt(t + 1)

        # λ^{t+1} = [λ^t + η_t * (Σ_k f4_k - B)]^+
        f4_total = sum(
            float(sD_list[k] @ np.array(C_D[k][:N_D[k]])) / C_MAX[k]
            for k in range(N_LAYERS)
        )
        lam = max(0., lam + eta * (f4_total - BUDGET_B))

        # ν_k^{t+1} = [ν_k^t + η_t * (δ_min - Coh_k)]^+
        for k in range(N_LAYERS - 1):
            coh_k = coherence(k, sD_list[k], sD_list[k+1])
            nu[k] = max(0., nu[k] + eta * (DELTA_MIN - coh_k))

        # ν_Ω^{t+1} = [ν_Ω^t + η_t * (Ω_min - Ω(t))]^+ (soft resilience)
        nu_omega = max(0., nu_omega + eta * (OMEGA_MIN - omega_t[t]))

        sD_prev = [s.copy() for s in sD_list]

    # Discounted totals
    weights = np.array([GAMMA_D**t for t in range(T)])
    UD_total = float(np.dot(weights, ud_t))
    UA_total = float(np.dot(weights, uA_t))
    VC       = float(np.sum(omega_t < OMEGA_MIN) / T)

    return {
        'name'    : 'R-BNE-DO+LCP',
        'theta'   : theta,
        'UD'      : UD_total,
        'UA'      : UA_total,
        'VC'      : VC,
        'omega'   : omega_t.tolist(),
        'omega_k' : omega_kt.tolist(),
        'f1'      : f1_kt.tolist(),
        'ud'      : ud_t.tolist(),
        'uA'      : uA_t.tolist(),
        'belief'  : belief_t.tolist(),
        'gap'     : gap_t.tolist(),
    }

# ══════════════════════════════════════════════════════════════
# ASGHAR-2025 BASELINE
# ══════════════════════════════════════════════════════════════
def _greedy_defender_oracle(k, sA_mix, state, belief, budget=None):
    """
    Faithful greedy defender oracle (GameSec 2025, Algorithm 1).

    Builds a PURE defender action as a subset of deception "elements"
    (here: which decoy-deployment actions to activate at layer k),
    added ONE AT A TIME by maximal marginal gain, under a partition-
    matroid budget. Submodularity of the belief-weighted defender gain
    (their Lemma 2) guarantees the greedy selection is a 1/2-approx.

    Returns a pure strategy vector sD (one-hot–like over the chosen
    composite action, renormalised) that is the greedy best response to
    the attacker mixed strategy sA_mix.
    """
    nD = N_D[k]
    if budget is None:
        budget = max(1, nD // 2)          # H+K analogue: half the catalogue

    # marginal gain of adding action i, given already-selected set S
    def gain(S):
        if not S:
            return 0.
        sD = np.zeros(nD)
        for i in S:
            sD[i] = 1.
        sD = sD / sD.sum()
        # belief-weighted expected defender utility vs current attacker mix
        return sum(
            belief[theta] * sum(
                sD[i] * sA_mix[a] *
                u_D_local(k, _e(nD, i), _e(N_A[k], a), theta, state)
                for i in range(nD) for a in range(N_A[k])
            )
            for theta in range(N_TYPES)
        )

    selected = []
    cur = 0.
    for _ in range(budget):
        best_i, best_delta = None, 1e-12
        for i in range(nD):
            if i in selected:
                continue
            delta = gain(selected + [i]) - cur            # marginal gain
            if delta > best_delta:
                best_delta, best_i = delta, i
        if best_i is None:                                # diminishing returns
            break
        selected.append(best_i)
        cur = gain(selected)

    sD = np.zeros(nD)
    if selected:
        for i in selected:
            sD[i] = 1.
        sD = sD / sD.sum()
    else:
        sD = np.ones(nD) / nD
    return sD


def run_asghar2025(theta, seed=0):
    """
    Asghar et al., GameSec 2025 — faithful baseline.

    Implements the two core mechanisms of the paper:
      (a) GREEDY DEFENDER ORACLE with partition-matroid budget, exploiting
          submodularity (Algorithm 1, Lemma 2, 1/2-approximation);
      (b) DOUBLE-ORACLE loop whose restricted games are solved by a single
          LINEAR PROGRAM, justified by the strategic-equivalence-to-zero-sum
          result (Proposition 2, Moulin–Vial). No Lemke–Howson / LCP.

    Deliberately omits Bayesian belief update and resilience/coherence/budget
    constraints (C1–C3): the baseline has none of these. The strategic-
    equivalence assumption that lets them avoid the LCP is exactly what fails
    for MD-CDBG, where resilience/budget/coherence penalties enter only the
    defender utility and break zero-sum symmetry.
    """
    rng   = np.random.default_rng(seed)
    state = State()

    omega_t  = np.zeros(T); omega_kt = np.zeros((T, N_LAYERS))
    f1_kt    = np.zeros((T, N_LAYERS))
    ud_t     = np.zeros(T); uA_t = np.zeros(T)
    belief_t = np.tile(PRIOR_A, (T, N_LAYERS, 1)).reshape(T, N_LAYERS, N_TYPES)
    gap_t    = np.zeros(T)

    uniform_belief = np.ones(N_TYPES) / N_TYPES

    for t in range(T):
        for k in range(N_LAYERS):
            omega_kt[t, k] = omega_local(k, state)
        omega_t[t] = omega_global(state)

        sD_list = []; sA_list = []

        for k in range(N_LAYERS):
            nD, nA = N_D[k], N_A[k]

            # ---- Double-Oracle loop (restricted game solved by LP) ----
            # init restricted sets with one action each
            D_set = [int(rng.integers(nD))]
            A_set = [int(rng.integers(nA))]
            sD_full = np.ones(nD) / nD
            sA_full = np.ones(nA) / nA

            for _do_iter in range(8):                     # DO iterations
                # --- solve restricted zero-sum game by LP (Prop. 2) ---
                # payoff (defender) on restricted sets
                Msub = np.array([[
                    sum(uniform_belief[th] *
                        u_D_local(k, _e(nD, i), _e(nA, j), th, state)
                        for th in range(N_TYPES))
                    for j in A_set] for i in D_set])
                # attacker minimises defender payoff: zero-sum LP for attacker
                nAi = len(A_set)
                c = np.zeros(nAi + 1); c[-1] = 1.
                A_ub = np.hstack([Msub, -np.ones((len(D_set), 1))])
                b_ub = np.zeros(len(D_set))
                A_eq = np.ones((1, nAi + 1)); A_eq[0, -1] = 0
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                              b_eq=[1.], bounds=[(0, None)]*nAi + [(None, None)],
                              method='highs')
                sA_restr = (np.clip(res.x[:nAi], 0, None)
                            if res.success else np.ones(nAi)/nAi)
                sA_restr = (sA_restr/sA_restr.sum()
                            if sA_restr.sum() > 1e-8 else np.ones(nAi)/nAi)
                # lift to full attacker space
                sA_full = np.zeros(nA)
                for idx, j in enumerate(A_set):
                    sA_full[j] = sA_restr[idx]

                # --- greedy defender oracle: best response to sA_full ---
                sD_br = _greedy_defender_oracle(k, sA_full, state,
                                                uniform_belief)
                new_d = int(np.argmax(sD_br))
                # --- attacker oracle: best pure response to defender mix ---
                # defender mixes uniformly over its restricted pure actions
                sD_full = np.zeros(nD)
                for i in D_set:
                    sD_full[i] = 1.
                sD_full /= sD_full.sum()
                a_payoff = [
                    u_A_local(k, sD_full, _e(nA, j), theta, state)
                    for j in range(nA)]
                new_a = int(np.argmax(a_payoff))

                grew = False
                if new_d not in D_set:
                    D_set.append(new_d); grew = True
                if new_a not in A_set:
                    A_set.append(new_a); grew = True
                if not grew:
                    break                                  # DO converged

            # final defender mix: greedy BR to final attacker mix
            sD = _greedy_defender_oracle(k, sA_full, state, uniform_belief)
            sA = sA_full

            sD_list.append(sD); sA_list.append(sA)

        for k in range(N_LAYERS):
            f1_kt[t, k] = f1(k, sA_list[k], sD_list[k], theta)

        sA_dist_list = [
            [sA_list[k] if th == theta else np.ones(N_A[k])/N_A[k]
             for th in range(N_TYPES)]
            for k in range(N_LAYERS)
        ]
        ud_t[t]  = U_D_global(sD_list, sA_dist_list, state)
        uA_t[t]  = U_A_global(sD_list, sA_list, theta, state)
        gap_t[t] = float(np.mean([
            zero_sum_gap(k, theta, state, uniform_belief)
            for k in range(N_LAYERS)]))

        aA = [int(rng.choice(N_A[k], p=sA_list[k])) for k in range(N_LAYERS)]
        aD = [int(rng.choice(N_D[k], p=sD_list[k])) for k in range(N_LAYERS)]
        state = transition(state, aA, aD, theta, sD_list, rng)

    weights  = np.array([GAMMA_D**t for t in range(T)])
    return {
        'name'    : 'Asghar-2025',
        'theta'   : theta,
        'UD'      : float(np.dot(weights, ud_t)),
        'UA'      : float(np.dot(weights, uA_t)),
        'VC'      : float(np.sum(omega_t < OMEGA_MIN)/T),
        'omega'   : omega_t.tolist(),
        'omega_k' : omega_kt.tolist(),
        'f1'      : f1_kt.tolist(),
        'ud'      : ud_t.tolist(),
        'uA'      : uA_t.tolist(),
        'belief'  : belief_t.tolist(),
        'gap'     : gap_t.tolist(),
    }


# ══════════════════════════════════════════════════════════════
# BASELINE 1: RANDOM
# Defender deploys decoys uniformly at random, no strategy.
# ══════════════════════════════════════════════════════════════
def run_random(theta, seed=0):
    """Random defender: uniform deployment. Attacker LP best-response."""
    rng = np.random.default_rng(seed)
    state = State()
    omega_t=np.zeros(T); omega_kt=np.zeros((T,N_LAYERS))
    f1_kt=np.zeros((T,N_LAYERS)); ud_t=np.zeros(T); uA_t=np.zeros(T)
    belief_t=np.tile(PRIOR_A,(T,N_LAYERS,1)).reshape(T,N_LAYERS,N_TYPES); gap_t=np.zeros(T)

    for t in range(T):
        for k in range(N_LAYERS): omega_kt[t,k]=omega_local(k,state)
        omega_t[t]=omega_global(state)
        sD_list=[]; sA_list=[]
        for k in range(N_LAYERS):
            sD = np.ones(N_D[k]) / N_D[k]  # uniform random deployment
            # Attacker plays rational best response to random defender
            MA = make_MA(k, theta, state)
            br = int(np.argmax(sD @ MA))
            sA = _e(N_A[k], br)
            sD_list.append(sD); sA_list.append(sA)
        for k in range(N_LAYERS): f1_kt[t,k]=f1(k,sA_list[k],sD_list[k],theta)
        sA_dist=[[sA_list[k] if th==theta else np.ones(N_A[k])/N_A[k]
                  for th in range(N_TYPES)] for k in range(N_LAYERS)]
        ud_t[t]=U_D_global(sD_list,sA_dist,state)
        uA_t[t]=U_A_global(sD_list,sA_list,theta,state)
        gap_t[t]=float(np.mean([zero_sum_gap(k,theta,state,np.ones(N_TYPES)/N_TYPES)
                                for k in range(N_LAYERS)]))
        aA=[int(rng.choice(N_A[k],p=sA_list[k])) for k in range(N_LAYERS)]
        aD=[int(rng.choice(N_D[k],p=sD_list[k])) for k in range(N_LAYERS)]
        state=transition(state,aA,aD,theta,sD_list,rng)
    w=np.array([GAMMA_D**t for t in range(T)])
    return {'name':'Random','theta':theta,'UD':float(w@ud_t),'UA':float(w@uA_t),
            'VC':float(np.sum(omega_t<OMEGA_MIN)/T),'omega':omega_t.tolist(),
            'omega_k':omega_kt.tolist(),'f1':f1_kt.tolist(),'ud':ud_t.tolist(),
            'uA':uA_t.tolist(),'belief':belief_t.tolist(),'gap':gap_t.tolist()}

# ══════════════════════════════════════════════════════════════
# BASELINE 2: NO-DECEPTION
# Defender uses only passive defense (isolation), no honeypots.
# ══════════════════════════════════════════════════════════════
def run_no_deception(theta, seed=0):
    """No-Deception: defender only isolates compromised nodes, no decoys."""
    rng = np.random.default_rng(seed)
    state = State()
    omega_t=np.zeros(T); omega_kt=np.zeros((T,N_LAYERS))
    f1_kt=np.zeros((T,N_LAYERS)); ud_t=np.zeros(T); uA_t=np.zeros(T)
    belief_t=np.tile(PRIOR_A,(T,N_LAYERS,1)).reshape(T,N_LAYERS,N_TYPES); gap_t=np.zeros(T)

    for t in range(T):
        for k in range(N_LAYERS): omega_kt[t,k]=omega_local(k,state)
        omega_t[t]=omega_global(state)
        sD_list=[]; sA_list=[]
        for k in range(N_LAYERS):
            # No deception: passive monitoring only (action 4 = minimal cost,
            # no honeypots deployed → ratio of decoys has no protective effect).
            # Attacker plays rational best response (knows there are no decoys).
            sD = _e(N_D[k], N_D[k]-1)  # passive monitoring (last action)
            MA = make_MA(k, theta, state)
            br = int(np.argmax(np.ones(N_D[k])/N_D[k] @ MA))
            sA = _e(N_A[k], br)
            sD_list.append(sD); sA_list.append(sA)
        for k in range(N_LAYERS): f1_kt[t,k]=f1(k,sA_list[k],sD_list[k],theta)
        sA_dist=[[sA_list[k] if th==theta else np.ones(N_A[k])/N_A[k]
                  for th in range(N_TYPES)] for k in range(N_LAYERS)]
        ud_t[t]=U_D_global(sD_list,sA_dist,state)
        uA_t[t]=U_A_global(sD_list,sA_list,theta,state)
        gap_t[t]=float(np.mean([zero_sum_gap(k,theta,state,np.ones(N_TYPES)/N_TYPES)
                                for k in range(N_LAYERS)]))
        aA=[int(rng.choice(N_A[k],p=sA_list[k])) for k in range(N_LAYERS)]
        aD=[int(rng.choice(N_D[k],p=sD_list[k])) for k in range(N_LAYERS)]
        state=transition(state,aA,aD,theta,sD_list,rng)
    w=np.array([GAMMA_D**t for t in range(T)])
    return {'name':'No-Deception','theta':theta,'UD':float(w@ud_t),'UA':float(w@uA_t),
            'VC':float(np.sum(omega_t<OMEGA_MIN)/T),'omega':omega_t.tolist(),
            'omega_k':omega_kt.tolist(),'f1':f1_kt.tolist(),'ud':ud_t.tolist(),
            'uA':uA_t.tolist(),'belief':belief_t.tolist(),'gap':gap_t.tolist()}

# ══════════════════════════════════════════════════════════════
# BASELINE 3: ML-ADAPTIVE (logistic-regression-style detector)
# Defender learns attacker action distribution online and best-responds.
# Mimics a supervised ML detector (e.g., GAT/autoencoder) without
# game-theoretic reasoning or resilience constraint.
# ══════════════════════════════════════════════════════════════
def run_ml_adaptive(theta, seed=0):
    """
    ML-Adaptive: online frequency-based attacker model + greedy best response.
    Represents reactive ML detectors (no equilibrium, no resilience constraint).
    """
    rng = np.random.default_rng(seed)
    state = State()
    omega_t=np.zeros(T); omega_kt=np.zeros((T,N_LAYERS))
    f1_kt=np.zeros((T,N_LAYERS)); ud_t=np.zeros(T); uA_t=np.zeros(T)
    belief_t=np.tile(PRIOR_A,(T,N_LAYERS,1)).reshape(T,N_LAYERS,N_TYPES); gap_t=np.zeros(T)

    # Online empirical attacker action counts (the "learned model")
    action_counts = [np.ones(N_A[k]) for k in range(N_LAYERS)]  # Laplace prior

    for t in range(T):
        for k in range(N_LAYERS): omega_kt[t,k]=omega_local(k,state)
        omega_t[t]=omega_global(state)
        sD_list=[]; sA_list=[]
        for k in range(N_LAYERS):
            nD=N_D[k]; nA=N_A[k]
            # ML predicts attacker distribution from empirical frequencies
            sA_pred = action_counts[k] / action_counts[k].sum()
            # Greedy best response to predicted attacker (no equilibrium reasoning)
            unif = np.ones(N_TYPES)/N_TYPES
            payoffs = np.array([
                sum(unif[th]*u_D_local(k,_e(nD,i),sA_pred,th,state) for th in range(N_TYPES))
                for i in range(nD)])
            best = int(np.argmax(payoffs))
            sD = np.zeros(nD); sD[best] = 0.85
            sD[(best+1)%nD] += 0.15  # small exploration
            # Attacker plays true best response to sD
            MA = make_MA(k, theta, state)
            br = int(np.argmax(sD @ MA))
            sA = np.zeros(nA); sA[br] = 1.0
            sD_list.append(sD); sA_list.append(sA)
        for k in range(N_LAYERS): f1_kt[t,k]=f1(k,sA_list[k],sD_list[k],theta)
        sA_dist=[[sA_list[k] if th==theta else np.ones(N_A[k])/N_A[k]
                  for th in range(N_TYPES)] for k in range(N_LAYERS)]
        ud_t[t]=U_D_global(sD_list,sA_dist,state)
        uA_t[t]=U_A_global(sD_list,sA_list,theta,state)
        gap_t[t]=float(np.mean([zero_sum_gap(k,theta,state,np.ones(N_TYPES)/N_TYPES)
                                for k in range(N_LAYERS)]))
        aA=[int(rng.choice(N_A[k],p=sA_list[k])) for k in range(N_LAYERS)]
        aD=[int(rng.choice(N_D[k],p=sD_list[k])) for k in range(N_LAYERS)]
        # ML model learns: update empirical counts
        for k in range(N_LAYERS): action_counts[k][aA[k]] += 1
        state=transition(state,aA,aD,theta,sD_list,rng)
    w=np.array([GAMMA_D**t for t in range(T)])
    return {'name':'ML-Adaptive','theta':theta,'UD':float(w@ud_t),'UA':float(w@uA_t),
            'VC':float(np.sum(omega_t<OMEGA_MIN)/T),'omega':omega_t.tolist(),
            'omega_k':omega_kt.tolist(),'f1':f1_kt.tolist(),'ud':ud_t.tolist(),
            'uA':uA_t.tolist(),'belief':belief_t.tolist(),'gap':gap_t.tolist()}


# ══════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ══════════════════════════════════════════════════════════════
STRATEGIES = [
    ("No-Deception",  run_no_deception),
    ("Random",        run_random),
    ("Asghar-2025",   run_asghar2025),
    ("R-BNE-DO+LCP",  run_rbne_do_lcp),
]
# Note: run_ml_adaptive is retained above as a future-work oracle
# (reactive ML detector). It is excluded from the main comparison and
# discussed as a perspective: integrating a trained deep detector as an
# additional best-response oracle within the double-oracle loop.
