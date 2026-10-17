"""Scalable Locally Injective Mappings, SLIM (Rabinovich, Poranne, Panozzo & Sorkine-Hornung 2017).

Free-boundary isometric map: minimizes the symmetric Dirichlet energy

    E = sum_t 2 A_t (s1^2 + s1^-2 + s2^2 + s2^-2),

with s1, s2 the singular values of each triangle's Jacobian (the factor 2 A_t
is libigl's double-area weight). Every iteration solves a reweighted
proxy, sum_t 2 A_t |W_t (J_t - R_t)|^2 plus a 1e-4 proximal term, where R_t is
the closest rotation to J_t and W_t = U diag(w) U^T with
w_i = sqrt(dD/ds_i / (2 (s_i - 1))) (w_i = 1 if |s_i - 1| < 1e-8). A line
search along the proxy direction starts below the first step that would
degenerate a triangle (0.8 of it, at most 1) and halves the step up to 12
times until the energy decreases, so the map stays locally injective. This
mirrors libigl's slim_solve. Initialization: harmonic map onto a circle (the
ARAP initialization), replaced by Tutte's embedding if it has flipped
triangles. Iterates until the relative energy decrease falls below tol.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .common import (boundary_loop, circle_boundary, cotan_laplacian, edge_lengths, face_areas_from_lengths,
                     gradient_operators, harmonic_extension, signed_areas_2d, uniform_laplacian)


def jacobians(Dx, Dy, UV):
    """(m, 2, 2) Jacobians [[du/dx, du/dy], [dv/dx, dv/dy]] in each triangle's local frame."""
    return np.stack([np.stack([Dx @ UV[:, 0], Dy @ UV[:, 0]], axis=1),
                     np.stack([Dx @ UV[:, 1], Dy @ UV[:, 1]], axis=1)], axis=1)


def symmetric_dirichlet(J, weights):
    """sum_t weights_t (|J|^2 + |J^-1|^2); +inf if any triangle is flipped or degenerate."""
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    if np.any(det <= 0):
        return np.inf
    frob = np.sum(J ** 2, axis=(1, 2))
    return float(np.sum(weights * frob * (1 + 1 / det ** 2)))


def initialization(V, F):
    n = len(V)
    b = boundary_loop(F)
    xb = circle_boundary(V, b)
    UV = harmonic_extension(cotan_laplacian(edge_lengths(V, F), F, n), b, xb)
    if np.any(signed_areas_2d(UV, F) <= 0):
        UV = harmonic_extension(uniform_laplacian(F, n), b, xb)  # Tutte: bijective for a convex boundary
    return UV


def _weights_and_rotations(J):
    U, s, Vt = np.linalg.svd(J)
    R = U @ Vt                                     # closest rotation (det J > 0 here)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.sqrt((s - s ** -3) / (s - 1))       # sqrt(dD/ds / (2 (s - 1))), D = s^2 + s^-2
    w[np.abs(s - 1) < 1e-8] = 1.0
    W = np.einsum("fik,fk,fjk->fij", U, w, U)
    return W, R


def _max_step(UV, d, F):
    """Smallest positive t at which some triangle of UV + t d degenerates (libigl's quadratic roots)."""
    U1, U2, U3 = UV[F[:, 0]], UV[F[:, 1]], UV[F[:, 2]]
    V1, V2, V3 = d[F[:, 0]], d[F[:, 1]], d[F[:, 2]]

    def cross(p, q):
        return p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]

    a = cross(V2 - V1, V3 - V1)
    b = cross(U2 - U1, V3 - V1) + cross(V2 - V1, U3 - U1)
    c = cross(U2 - U1, U3 - U1)
    t = np.full(len(F), np.inf)
    quad = np.abs(a) > 1e-10
    disc = b ** 2 - 4 * a * c
    ok = quad & (disc > 0)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    # Numerically stable roots, as in libigl's get_smallest_pos_quad_zero.
    bd = np.where(b >= 0, -b - sq, -b + sq)
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.where(b >= 0, 2 * c / bd, bd / (2 * a))
        r2 = np.where(b >= 0, bd / (2 * a), 2 * c / bd)
    hi, lo = np.where(a < 0, r2, r1), np.where(a < 0, r1, r2)   # hi >= lo
    t[ok] = np.where(hi[ok] > 0, np.where(lo[ok] > 0, lo[ok], hi[ok]), np.inf)
    lin = ~quad & (b != 0)
    with np.errstate(divide="ignore"):
        tl = -c[lin] / b[lin]
    t[lin] = np.where(tl > 0, tl, np.inf)
    return t.min()


def flatten(V, F, tol=1e-10, max_iter=5000, UV0=None, fixed_iterations=None):
    """fixed_iterations runs exactly that many iterations (used to compare with libigl)."""
    n, m = len(V), len(F)
    Dx, Dy = gradient_operators(V, F)
    M = 2 * face_areas_from_lengths(edge_lengths(V, F))       # double areas, as libigl
    UV = initialization(V, F) if UV0 is None else UV0.copy()
    energy = symmetric_dirichlet(jacobians(Dx, Dy, UV), M)
    proximal = 1e-4
    Mw = np.tile(M, 4)

    iterations = max_iter if fixed_iterations is None else fixed_iterations
    for it in range(iterations):
        W, R = _weights_and_rotations(jacobians(Dx, Dy, UV))
        # A x = vec(W J(x)) with x = [u; v]; blocks (WJ)_00, (WJ)_01, (WJ)_10, (WJ)_11.
        A = sp.vstack([sp.hstack([sp.diags(W[:, i, 0]) @ D, sp.diags(W[:, i, 1]) @ D])
                       for i in range(2) for D in (Dx, Dy)]).tocsr()
        WR = np.einsum("fij,fjk->fik", W, R)
        r = np.concatenate([WR[:, 0, 0], WR[:, 0, 1], WR[:, 1, 0], WR[:, 1, 1]])
        x0 = np.concatenate([UV[:, 0], UV[:, 1]])
        L = (A.T @ sp.diags(Mw) @ A + proximal * sp.identity(2 * n)).tocsc()
        x = spla.spsolve(L, A.T @ (Mw * r) + proximal * x0)
        d = np.stack([x[:n], x[n:]], axis=1) - UV

        step = min(1.0, 0.8 * _max_step(UV, d, F))
        new_energy = energy
        for _ in range(12):
            trial = UV + step * d
            e = symmetric_dirichlet(jacobians(Dx, Dy, trial), M)
            if e < energy:
                UV, new_energy = trial, e
                break
            step /= 2
        decrease = (energy - new_energy) / energy
        energy = new_energy
        if fixed_iterations is None and decrease < tol:
            return UV
    if fixed_iterations is None:
        raise RuntimeError(f"SLIM did not converge in {max_iter} iterations")
    return UV
