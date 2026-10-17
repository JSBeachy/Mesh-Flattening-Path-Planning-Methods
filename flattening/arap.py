"""As-Rigid-As-Possible parameterization (Liu et al. 2008), paper Sec. 2.1.3.

Energy: E = sum_t sum_{(i,j) in t} cot(theta_ij) |(u_i - u_j) - R_t (x_i - x_j)|^2,
where x is the isometric 2D layout of each triangle and theta_ij the angle
opposite edge (i, j). Minimized with the local/global scheme: the local step
picks the rotation R_t closest to each triangle's Jacobian, the global step
solves a cotangent-Laplacian system. Initialized with a harmonic map whose
boundary lies on a circle; iterated until the largest vertex displacement
between two iterations falls below tol * (mean edge length).
"""
import numpy as np
import scipy.sparse.linalg as spla

from .common import (boundary_loop, circle_boundary, cotan_laplacian, cotangents, edge_lengths,
                     harmonic_extension, local_frames)


def harmonic_initialization(V, F, L):
    """Harmonic map with the boundary on a circle, spaced by arc length."""
    b = boundary_loop(F)
    return harmonic_extension(L, b, circle_boundary(V, b))


def flatten(V, F, tol=1e-6, max_iter=10000):
    n = len(V)
    l = edge_lengths(V, F)
    cot = cotangents(l)
    L = cotan_laplacian(l, F, n).tocsc()
    X = local_frames(V, F)
    # Inverse of each triangle's 2x2 edge matrix [x1 - x0, x2 - x0] (upper triangular).
    Xinv = np.zeros((len(F), 2, 2))
    Xinv[:, 0, 0] = 1 / X[:, 1, 0]
    Xinv[:, 0, 1] = -X[:, 2, 0] / (X[:, 1, 0] * X[:, 2, 1])
    Xinv[:, 1, 1] = 1 / X[:, 2, 1]

    U = harmonic_initialization(V, F, L)
    free = np.arange(1, n)  # vertex 0 stays fixed to remove the translation null space
    solve = spla.factorized(L[free][:, free].tocsc())
    L_f0 = L[free][:, [0]].toarray().ravel()
    tol_abs = tol * l.mean()

    for _ in range(max_iter):
        # Local step: R_t = rotation closest to the Jacobian J_t (maximizes trace(R^T J)).
        D = np.stack([U[F[:, 1]] - U[F[:, 0]], U[F[:, 2]] - U[F[:, 0]]], axis=2)
        J = D @ Xinv
        ang = np.arctan2(J[:, 1, 0] - J[:, 0, 1], J[:, 0, 0] + J[:, 1, 1])
        c, s = np.cos(ang), np.sin(ang)
        R = np.stack([np.stack([c, -s], axis=1), np.stack([s, c], axis=1)], axis=1)

        # Global step: L u = sum over triangle edges of 0.5 cot(theta) R_t (x_i - x_j).
        rhs = np.zeros((n, 2))
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            e = 0.5 * cot[:, k, None] * np.einsum("fab,fb->fa", R, X[:, i] - X[:, j])
            for d in range(2):
                rhs[:, d] += (np.bincount(F[:, i], e[:, d], minlength=n)
                              - np.bincount(F[:, j], e[:, d], minlength=n))
        U_new = U.copy()
        for d in range(2):
            U_new[free, d] = solve(rhs[free, d] - L_f0 * U[0, d])
        step = np.abs(U_new - U).max()
        U = U_new
        if step < tol_abs:
            break
    return U
