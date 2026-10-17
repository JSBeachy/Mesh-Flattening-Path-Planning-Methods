"""ABF++: Angle Based Flattening (Sheffer, Levy, Mogilnitsky & Bogomyakov 2005).

Free-boundary conformal map computed in angle space. The unknowns are the 3m
corner angles alpha of the flat triangulation; they minimize

    E(alpha) = sum w (alpha - beta)^2,   w = 1 / beta^2,

where beta are the 3D angles, scaled to sum to 2 pi around interior vertices
(and clamped to [3, 175] degrees), subject to
    triangle:     alpha_1 + alpha_2 + alpha_3 = pi                  (each face),
    planarity:    sum of the angles around v = 2 pi                  (interior v),
    wheel:        prod sin(alpha_next) = prod sin(alpha_prev)        (interior v).

Solved by sequential linearly constrained programming (Newton's method on the
Lagrangian without the second derivatives of the constraints) with the two
ABF++ Schur-complement reductions, so that only a 2 n_int system is factored
per iteration. Settings follow the authors' implementation in Geogram (target
angles, weights, step-length control); we iterate to a tight tolerance on the
KKT residual instead of Geogram's loose one. The (u, v) coordinates are then
recovered from the angles with the angle-based least-squares formulation used
by ABF++: (U2 - U0) = sin(a1) / sin(a2) exp(i a0) (U1 - U0) per triangle,
rows weighted by sqrt(3D area), two vertices locked (the extremes along the
longest bounding-box axis). With converged angles the recovery is exact.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .common import boundary_loop, corner_angles, edge_lengths, face_areas_from_lengths


def target_angles(V, F):
    """beta: 3D corner angles, scaled to 2 pi around interior vertices and clamped (as in Geogram)."""
    n = len(V)
    theta = np.maximum(corner_angles(edge_lengths(V, F)), np.radians(1.0))
    interior = np.ones(n, dtype=bool)
    interior[boundary_loop(F)] = False
    ratio = np.where(interior, 2 * np.pi / np.bincount(F.ravel(), theta.ravel(), minlength=n), 1.0)
    return np.clip(theta * ratio[F], np.radians(3.0), np.radians(175.0)), interior


def solve_angles(V, F, tol=1e-10, max_iter=100, eps=1e-5):
    """Optimal flat angles (m, 3); raises RuntimeError if the iteration does not converge."""
    m = len(F)
    beta, interior = target_angles(V, F)
    beta, w = beta.ravel(), 1.0 / beta.ravel() ** 2
    alpha = beta.copy()
    corner_vertex = F.ravel()
    nxt = (np.arange(3 * m) // 3) * 3 + (np.arange(3 * m) + 1) % 3   # next corner in the same face
    prv = (np.arange(3 * m) // 3) * 3 + (np.arange(3 * m) + 2) % 3   # previous corner
    vid = -np.ones(len(V), dtype=np.int64)
    vid[interior] = np.arange(interior.sum())
    k = int(interior.sum())
    inner = interior[corner_vertex]                  # corners at interior vertices
    row = vid[corner_vertex[inner]]                  # their constraint row
    J1 = sp.csr_matrix((np.ones(3 * m), (np.arange(3 * m) // 3, np.arange(3 * m))), shape=(m, 3 * m))
    lam_tri, lam = np.zeros(m), np.zeros(2 * k)

    for _ in range(max_iter):
        # Constraints and their Jacobian J2 = [planarity; wheel] (2k x 3m).
        log_sin = np.log(np.sin(alpha))
        p_next = np.exp(np.bincount(row, log_sin[nxt[inner]], minlength=k))
        p_prev = np.exp(np.bincount(row, log_sin[prv[inner]], minlength=k))
        cot = np.cos(alpha) / np.sin(alpha)
        J2 = sp.csr_matrix((np.concatenate([np.ones(inner.sum()), p_next[row] * cot[nxt[inner]],
                                            -p_prev[row] * cot[prv[inner]]]),
                            (np.concatenate([row, k + row, k + row]),
                             np.concatenate([np.flatnonzero(inner), nxt[inner], prv[inner]]))),
                           shape=(2 * k, 3 * m))
        C_tri = J1 @ alpha - np.pi
        C2 = np.concatenate([np.bincount(row, alpha[inner], minlength=k) - 2 * np.pi, p_next - p_prev])
        b1 = -(2 * w * (alpha - beta) + J1.T @ lam_tri + J2.T @ lam)   # -grad_alpha of the Lagrangian
        b2_tri, b2 = -C_tri, -C2
        if max(np.abs(b1).max(), np.abs(b2_tri).max(), np.abs(b2).max()) < tol:
            return alpha.reshape(m, 3)

        # One SLCP step: [Lambda J^T; J 0] [d_alpha; d_lambda] = [b1; b2], Lambda = diag(2 w),
        # reduced twice (Sheffer et al., Eqs. 9-17).
        D = 1.0 / (2 * w)
        D_star = 1.0 / (J1 @ D)                                    # (J1 D J1^T)^-1, diagonal
        J_star = J2 @ sp.diags(D) @ J1.T                           # J2 D J1^T
        b1_star = J1 @ (D * b1) - b2_tri
        b2_star = J2 @ (D * b1) - b2
        S = (J_star @ sp.diags(D_star) @ J_star.T - J2 @ sp.diags(D) @ J2.T).tocsc()
        d_lam = spla.spsolve(S, J_star @ (D_star * b1_star) - b2_star)
        d_lam_tri = D_star * (b1_star - J_star.T @ d_lam)
        d_alpha = D * (b1 - J1.T @ d_lam_tri - J2.T @ d_lam)

        # Step length control from Geogram: keep the angles positive, stiffen angles near 0 or pi.
        s = 1.0
        low = alpha + d_alpha < 10 * eps
        if low.any():
            s = min(s, np.min(-0.5 * (alpha[low] - 10 * eps) / d_alpha[low]))
        w[low | (alpha + d_alpha > np.pi - 10 * eps)] *= 1.2
        alpha += s * d_alpha
        lam_tri += s * d_lam_tri
        lam += s * d_lam
        if not np.all(np.isfinite(alpha)):
            break
    raise RuntimeError("ABF++ did not converge")


def angles_to_uv(V, F, alpha):
    """Least-squares (u, v) from triangle angles (exact when the angles are consistent)."""
    n, m = len(V), len(F)
    s = np.sin(alpha[:, 1]) / np.sin(alpha[:, 2])
    a, b = s * np.cos(alpha[:, 0]), s * np.sin(alpha[:, 0])
    weight = np.sqrt(face_areas_from_lengths(edge_lengths(V, F)))
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    # Unknowns x = [u; v]. Real part:  (1-a) u0 + b v0 + a u1 - b v1 - u2 = 0
    #                      Imaginary: -b u0 + (1-a) v0 + b u1 + a v1 - v2 = 0
    one = np.ones(m)
    re_cols = np.stack([i0, i0 + n, i1, i1 + n, i2], axis=1)
    im_cols = np.stack([i0, i0 + n, i1, i1 + n, i2 + n], axis=1)
    re_vals = np.stack([1 - a, b, a, -b, -one], axis=1) * weight[:, None]
    im_vals = np.stack([-b, 1 - a, b, a, -one], axis=1) * weight[:, None]
    rows = np.repeat(np.arange(2 * m), 5)
    A = sp.csr_matrix((np.concatenate([re_vals, im_vals]).ravel(),
                       (rows, np.concatenate([re_cols, im_cols]).ravel())), shape=(2 * m, 2 * n))

    # Lock the extreme vertices along the longest bounding-box axis at their projections
    # onto the two longest axes (as Geogram's LSCM does).
    axes = np.argsort(V.max(axis=0) - V.min(axis=0))[::-1][:2]
    p, r = np.argmin(V[:, axes[0]]), np.argmax(V[:, axes[0]])
    locked = np.array([p, r, p + n, r + n])
    x_locked = np.array([V[p, axes[0]], V[r, axes[0]], V[p, axes[1]], V[r, axes[1]]])
    free = np.setdiff1d(np.arange(2 * n), locked)
    A = A.tocsc()
    Af = A[:, free]
    x = np.zeros(2 * n)
    x[locked] = x_locked
    x[free] = spla.spsolve((Af.T @ Af).tocsc(), -(Af.T @ (A[:, locked] @ x_locked)))
    return np.stack([x[:n], x[n:]], axis=1)


def flatten(V, F):
    return angles_to_uv(V, F, solve_angles(V, F))
