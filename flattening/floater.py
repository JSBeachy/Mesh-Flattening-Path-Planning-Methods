"""Mean value coordinates parameterization (Floater 2003), a fixed-boundary baseline.

The boundary is fixed to a circle (spaced by arc length) and every interior
vertex is a convex combination of its neighbors with the mean value weights

    w_ij = (tan(delta_ij / 2) + tan(gamma_ij / 2)) / |x_j - x_i|,

where delta_ij and gamma_ij are the 3D angles at x_i in the two triangles that
share the edge (i, j). The weights are positive, so by Tutte's theorem the map
is bijective. Unlike the free-boundary methods it cannot adapt the boundary
shape, which is what the paper's choice of free-boundary methods avoids.
"""
import numpy as np
import scipy.sparse as sp

from .common import boundary_loop, circle_boundary, corner_angles, edge_lengths, harmonic_extension


def mean_value_laplacian(V, F):
    """Non-symmetric matrix with L_ij = -w_ij and L_ii = sum_j w_ij."""
    l = edge_lengths(V, F)
    t = np.tan(0.5 * corner_angles(l))
    rows, cols, vals = [], [], []
    for c in range(3):
        i, j, k = F[:, c], F[:, (c + 1) % 3], F[:, (c + 2) % 3]
        # Corner c contributes tan(angle_c / 2) to the weights of its two edges (i, j) and (i, k);
        # edge (i, j) is opposite corner k = c + 2, edge (i, k) opposite corner j = c + 1.
        for other, length in ((j, l[:, (c + 2) % 3]), (k, l[:, (c + 1) % 3])):
            rows.append(i)
            cols.append(other)
            vals.append(t[:, c] / length)
    W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(V), len(V)))
    return sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W


def flatten(V, F):
    b = boundary_loop(F)
    return harmonic_extension(mean_value_laplacian(V, F), b, circle_boundary(V, b))
