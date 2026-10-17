"""Spectral Conformal Parameterization (Mullen, Tong, Alliez & Desbrun 2008).

Free-boundary conformal map without pinned vertices. With L_C the conformal
energy matrix (the quadratic form that LSCM minimizes) and B the diagonal
matrix with ones at the boundary vertices, SCP is the generalized eigenvector

    L_C u = lambda B u    with the smallest non-trivial lambda    (paper Eq. 5),

i.e. the map minimizing the conformal energy for a unit boundary moment of
inertia about the boundary barycenter (Eq. 6). As in the paper (Eq. 7), it is
computed as the eigenvector with the largest eigenvalue mu of

    (B - e_b e_b^T / V_b) u = mu (L_C + eps B) u,

where the rank-2 term removes the constant (translation) solutions. The paper
adds eps I (eps = 1e-8) to keep the right-hand matrix positive definite; that
perturbs the eigenvector slightly (on our animal head the mean area distortion
changes by 0.7 %). Adding eps B instead also makes it positive definite (for
developable meshes too) but leaves the eigenvectors exactly unchanged: for
boundary-centered u, B u = mu (L_C + eps B) u is L_C u = (1 / mu - eps) B u.
The map is determined up to a similarity; we rotate it so that the boundary
pair LSCM pins lies along +x, which makes the orientation deterministic.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .lscm import conformal_matrix, farthest_boundary_pair


def flatten(V, F, eps=1e-8, tol=1e-12):
    n = len(V)
    Q, b = conformal_matrix(V, F)
    on_boundary = np.zeros(2 * n)
    on_boundary[b] = on_boundary[b + n] = 1.0
    e_b = np.zeros((2 * n, 2))
    e_b[b, 0] = e_b[b + n, 1] = 1.0
    M = (Q + eps * sp.diags(on_boundary)).tocsc()

    A = spla.LinearOperator((2 * n, 2 * n), dtype=float,
                            matvec=lambda x: on_boundary * x.ravel() - e_b @ (e_b.T @ x.ravel()) / len(b))
    lu = spla.splu(M)
    Minv = spla.LinearOperator((2 * n, 2 * n), dtype=float, matvec=lambda x: lu.solve(x.ravel()))
    v0 = np.ones(2 * n) + np.arange(2 * n) / (2 * n)  # fixed start vector: deterministic result
    _, vec = spla.eigsh(A, k=1, M=M, Minv=Minv, which="LA", v0=v0, tol=tol, maxiter=100000)
    UV = np.stack([vec[:n, 0], vec[n:, 0]], axis=1)

    p, r = farthest_boundary_pair(V, b)
    d = UV[r] - UV[p]
    c, s = d / np.linalg.norm(d)
    return (UV - UV[p]) @ np.array([[c, -s], [s, c]])
