"""Conformal Equivalence of Triangle Meshes (Springborn, Schroeder & Pinkall 2008), paper Sec. 2.1.4.

Discrete conformal scaling of edge lengths, l_ij = exp((u_i + u_j) / 2) l0_ij.
Free-boundary flattening: interior vertices target the angle sum 2*pi (flat),
boundary vertices keep u = 0 (their edge lengths are preserved). The convex energy

    E(u) = sum_t [2 f_t - pi/2 (lam_1 + lam_2 + lam_3)] + sum_i ThetaHat_i u_i,
    f_t  = 1/2 sum_k theta_k lam_k + sum_k Lob(theta_k),    lam = 2 log l,

has gradient ThetaHat - Theta(u) (target minus current angle sums) and the
cotangent Laplacian of the current metric as Hessian. It is minimized with
Newton's method and a backtracking line search that only accepts steps keeping
every triangle inequality satisfied (without intrinsic edge flips, meshes with
very obtuse triangles can therefore fail; the paper remeshes in that case,
and so do we, see make_meshes.py). Finally the boundary polygon
is built from the boundary lengths and exterior angles of the flat metric and
extended to the interior harmonically with that metric's cotangent weights,
which reproduces the developed flat triangles exactly (linear precision).
"""
import numpy as np
import scipy.sparse.linalg as spla
from scipy.special import zeta

from .common import (boundary_curve, boundary_loop, corner_angles, cotan_laplacian, edge_lengths,
                     harmonic_extension, vertex_angle_sums)

_N = np.arange(1, 31)
_CLAUSEN_COEF = zeta(2 * _N) / ((2 * np.pi) ** (2 * _N) * _N * (2 * _N + 1))  # = |B_2n| / (2n (2n+1)!)


def clausen2(x):
    """Clausen function Cl_2(x) = -int_0^x log|2 sin(t/2)| dt.

    Uses Cl_2(x) = x - x log|x| + sum_n |B_2n| x^(2n+1) / (2n (2n+1)!) on [-pi, pi]
    (2*pi-periodic, convergent for |x| < 2*pi).
    """
    x = np.mod(np.asarray(x, dtype=float) + np.pi, 2 * np.pi) - np.pi
    y = x * x
    p = np.zeros_like(x)
    for c in _CLAUSEN_COEF[::-1]:
        p = p * y + c
    ax = np.abs(x)
    return x - x * np.log(np.where(ax > 0, ax, 1.0)) + x * y * p


def lobachevsky(theta):
    """Lobachevsky function Lob(theta) = -int_0^theta log|2 sin t| dt = Cl_2(2 theta) / 2."""
    return 0.5 * clausen2(2 * theta)


def _log_lengths(u, lam0, F):
    """lam = 2 log l of the scaled metric for the edge opposite each corner."""
    return lam0 + u[F[:, [1, 2, 0]]] + u[F[:, [2, 0, 1]]]


def energy(u, lam0, F, theta_hat):
    lam = _log_lengths(u, lam0, F)
    theta = corner_angles(np.exp(0.5 * lam))
    f = 0.5 * np.sum(theta * lam, axis=1) + np.sum(lobachevsky(theta), axis=1)
    return np.sum(2 * f - 0.5 * np.pi * np.sum(lam, axis=1)) + theta_hat @ u


def solve_scale_factors(V, F, tol=1e-10, max_iter=100):
    """Newton's method for the log scale factors u (u = 0 on the boundary)."""
    n = len(V)
    lam0 = 2 * np.log(edge_lengths(V, F))
    interior = np.setdiff1d(np.arange(n), boundary_loop(F))
    theta_hat = np.zeros(n)
    theta_hat[interior] = 2 * np.pi  # boundary entries are irrelevant because u_b is fixed

    u = np.zeros(n)
    E = energy(u, lam0, F, theta_hat)
    for _ in range(max_iter):
        l = np.exp(0.5 * _log_lengths(u, lam0, F))
        grad = (theta_hat - vertex_angle_sums(l, F, n))[interior]
        if np.abs(grad).max() < tol:
            return u
        H = cotan_laplacian(l, F, n)[interior][:, interior]
        d = spla.spsolve(H.tocsc(), -grad)
        decrement = -grad @ d  # Newton decrement squared (> 0)
        t = 1.0
        while True:  # backtracking line search keeping all triangles valid (Armijo condition;
            u_new = u.copy()  # full steps once the decrement is below round-off level)
            u_new[interior] += t * d
            l_new = np.exp(0.5 * _log_lengths(u_new, lam0, F))
            if np.all(2 * l_new < l_new.sum(axis=1, keepdims=True)):  # strict triangle inequalities
                E_new = energy(u_new, lam0, F, theta_hat)
                if E_new <= E - 1e-4 * t * decrement or decrement < 1e-9:
                    break
            t *= 0.5
            if t < 1e-12:
                raise RuntimeError("CETM line search failed")
        u, E = u_new, E_new
    raise RuntimeError(f"CETM Newton did not converge in {max_iter} iterations "
                       f"(max angle error {np.abs(grad).max():.2e})")


def flatten(V, F):
    n = len(V)
    u = solve_scale_factors(V, F)
    l = np.exp(0.5 * _log_lengths(u, 2 * np.log(edge_lengths(V, F)), F))
    b = boundary_loop(F)
    b_next = np.roll(b, -1)
    lengths = np.linalg.norm(V[b_next] - V[b], axis=1) * np.exp(0.5 * (u[b] + u[b_next]))
    uv_boundary = boundary_curve(lengths, np.pi - vertex_angle_sums(l, F, n)[b])
    return harmonic_extension(cotan_laplacian(l, F, n), b, uv_boundary)
