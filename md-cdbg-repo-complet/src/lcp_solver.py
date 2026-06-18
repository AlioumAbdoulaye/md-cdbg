"""
lcp_solver.py — Nash Equilibrium for non-zero-sum bimatrix games
via support enumeration (exact, feasible for small restricted sets).

For the restricted sets R_D x R_A in Bayes-DO, typical sizes are
2-6 actions per player → support enumeration is tractable.
"""

import numpy as np
from itertools import combinations
from scipy.optimize import linprog


def solve_bimatrix_support_enum(M_D, M_A, tol=1e-8):
    """
    Exact Nash Equilibrium of bimatrix game (M_D, M_A)
    via support enumeration (Porter et al., 2004).

    M_D[i,j] = defender payoff when defender plays i, attacker plays j
    M_A[i,j] = attacker payoff when defender plays i, attacker plays j

    Returns (sigma_D, sigma_A) — mixed NE strategies.
    Falls back to minimax LP if no NE found (shouldn't happen for finite games).
    """
    nD, nA = M_D.shape
    assert M_A.shape == (nD, nA), "M_D and M_A must have same shape"

    best = None
    best_val = -np.inf   # maximize defender utility at NE

    # Enumerate all pairs of supports (S_D, S_A)
    for s_D_size in range(1, nD + 1):
        for s_A_size in range(1, nA + 1):
            for S_D in combinations(range(nD), s_D_size):
                for S_A in combinations(range(nA), s_A_size):
                    sol = _check_support(M_D, M_A, S_D, S_A, tol)
                    if sol is not None:
                        sigma_D, sigma_A = sol
                        val = float(sigma_D @ M_D @ sigma_A)
                        # Keep the NE with highest defender value
                        # (Nash equilibrium selection: maximal for defender)
                        if val > best_val:
                            best_val = val
                            best = (sigma_D, sigma_A)

    if best is not None:
        return best

    # Fallback: minimax LP (should not be reached for finite games)
    return _minimax_lp_fallback(M_D, nD, nA)


def _check_support(M_D, M_A, S_D, S_A, tol):
    """
    Check if (S_D, S_A) can be a NE support.
    NE conditions:
      - For each i in S_D: sum_j sigma_A[j] * M_D[i,j] = v_D  (indifference)
      - For each i not in S_D: sum_j sigma_A[j] * M_D[i,j] <= v_D
      - For each j in S_A: sum_i sigma_D[i] * M_A[i,j] = v_A  (indifference)
      - For each j not in S_A: sum_i sigma_D[i] * M_A[i,j] <= v_A
      - sigma_D[i] >= 0, sum = 1; sigma_A[j] >= 0, sum = 1
    """
    nD, nA = M_D.shape
    S_D = list(S_D); S_A = list(S_A)
    nSD = len(S_D); nSA = len(S_A)

    # ── Solve for sigma_A given support S_A ──────────────────
    # Indifference for defender over S_D:
    # M_D[i, S_A] · sigma_A[S_A] = v_D  for all i in S_D
    # sum(sigma_A[S_A]) = 1
    # sigma_A[S_A] >= 0

    if nSA == 1:
        sigma_A = np.zeros(nA)
        sigma_A[S_A[0]] = 1.0
    else:
        # Build system: M_D[S_D, :][:, S_A] * x = v_D * ones
        # Augment: last eq is sum(x) = 1
        # Variables: [x_1,...,x_nSA, v_D]
        A = np.zeros((nSD, nSA + 1))
        for row, i in enumerate(S_D):
            A[row, :nSA] = M_D[i, S_A]
            A[row, nSA]  = -1.0   # -v_D
        b = np.zeros(nSD)

        # Add sum constraint
        A_sum = np.zeros((1, nSA + 1))
        A_sum[0, :nSA] = 1.0
        b_sum = np.array([1.0])

        A_full = np.vstack([A, A_sum])
        b_full = np.concatenate([b, b_sum])

        # Solve least-squares
        try:
            sol, res, rank, sv = np.linalg.lstsq(A_full, b_full, rcond=None)
        except np.linalg.LinAlgError:
            return None

        x = sol[:nSA]
        if np.any(x < -tol):
            return None
        x = np.clip(x, 0, None)
        if abs(x.sum()) < tol:
            return None
        x /= x.sum()
        sigma_A = np.zeros(nA)
        for j, idx in enumerate(S_A):
            sigma_A[idx] = x[j]

    # ── Verify defender indifference + no profitable deviation ─
    payoffs_D = M_D @ sigma_A
    v_D = float(np.mean(payoffs_D[S_D]))
    for i in S_D:
        if abs(payoffs_D[i] - v_D) > tol * 10:
            return None
    for i in range(nD):
        if i not in S_D and payoffs_D[i] > v_D + tol:
            return None

    # ── Solve for sigma_D given support S_D ──────────────────
    if nSD == 1:
        sigma_D = np.zeros(nD)
        sigma_D[S_D[0]] = 1.0
    else:
        A2 = np.zeros((nSA, nSD + 1))
        for row, j in enumerate(S_A):
            A2[row, :nSD] = M_A[S_D, j]
            A2[row, nSD]  = -1.0
        b2 = np.zeros(nSA)
        A2_sum = np.zeros((1, nSD + 1))
        A2_sum[0, :nSD] = 1.0
        b2_sum = np.array([1.0])
        A2_full = np.vstack([A2, A2_sum])
        b2_full = np.concatenate([b2, b2_sum])

        try:
            sol2, _, _, _ = np.linalg.lstsq(A2_full, b2_full, rcond=None)
        except np.linalg.LinAlgError:
            return None

        y = sol2[:nSD]
        if np.any(y < -tol):
            return None
        y = np.clip(y, 0, None)
        if abs(y.sum()) < tol:
            return None
        y /= y.sum()
        sigma_D = np.zeros(nD)
        for i, idx in enumerate(S_D):
            sigma_D[idx] = y[i]

    # ── Verify attacker indifference + no profitable deviation ─
    payoffs_A = sigma_D @ M_A
    v_A = float(np.mean(payoffs_A[S_A]))
    for j in S_A:
        if abs(payoffs_A[j] - v_A) > tol * 10:
            return None
    for j in range(nA):
        if j not in S_A and payoffs_A[j] > v_A + tol:
            return None

    return sigma_D, sigma_A


def _minimax_lp_fallback(M_D, nD, nA):
    """Fallback minimax LP if support enumeration fails."""
    c = np.zeros(nD + 1); c[-1] = -1.0
    A_ub = np.hstack([-M_D.T, np.ones((nA, 1))])
    b_ub = np.zeros(nA)
    A_eq = np.ones((1, nD + 1)); A_eq[0, -1] = 0
    b_eq = np.array([1.0])
    bounds = [(0, None)] * nD + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    sD = np.ones(nD) / nD
    if res.success:
        sD = np.clip(res.x[:nD], 0, None)
        sD = sD / sD.sum() if sD.sum() > 1e-8 else np.ones(nD) / nD
    sA = np.ones(nA) / nA
    return sD, sA
