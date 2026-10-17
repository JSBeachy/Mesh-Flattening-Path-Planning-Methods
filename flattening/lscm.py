"""Least-Squares Conformal Maps (Levy et al. 2002), free boundary (paper Sec. 2.1.1).

The LSCM energy equals the conformal energy E_C = E_D - A (Mullen et al. 2008):
the Dirichlet energy E_D = 1/2 (u^T L u + v^T L v) minus the signed area A of
the map. Written as E_C = 1/2 x^T (L (x) I_2 - 2 A_vec) x with x = [u; v], this
is the sparse system quoted in the paper. Two boundary vertices are pinned to
(0, 0) and (1, 0) to remove the similarity null space.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .common import boundary_loop, cotan_laplacian, edge_lengths


def flatten(V, F):
    n = len(V)
    L = cotan_laplacian(edge_lengths(V, F), F, n)

    # Signed area A = 1/2 sum over boundary edges (i -> j) of (u_i v_j - u_j v_i),
    # written as the symmetric quadratic form x^T A_vec x.
    b = boundary_loop(F)
    i, j = b, np.roll(b, -1)
    q = np.full(len(b), 0.25)
    A_vec = sp.csr_matrix((np.concatenate([q, q, -q, -q]),
                           (np.concatenate([i, j + n, j, i + n]),
                            np.concatenate([j + n, i, i + n, j]))), shape=(2 * n, 2 * n))
    Q = (sp.block_diag([L, L]) - 2 * A_vec).tocsr()

    # Pin the two boundary vertices that are farthest apart.
    D = np.linalg.norm(V[b][:, None, :] - V[b][None, :, :], axis=2)
    p, r = np.unravel_index(np.argmax(D), D.shape)
    pinned = np.array([b[p], b[r], b[p] + n, b[r] + n])
    x_pinned = np.array([0.0, 1.0, 0.0, 0.0])
    free = np.setdiff1d(np.arange(2 * n), pinned)

    x = np.zeros(2 * n)
    x[pinned] = x_pinned
    x[free] = spla.spsolve(Q[free][:, free].tocsc(), -(Q[free][:, pinned] @ x_pinned))
    return np.stack([x[:n], x[n:]], axis=1)
