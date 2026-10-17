"""Boundary First Flattening (Sawhney & Crane 2017), paper Sec. 2.1.2.

Minimum-distortion configuration: the boundary log scale factors are u_b = 0,
i.e. boundary lengths are preserved. Steps:
  1. Dirichlet-to-Neumann (Poincare-Steklov) map: solve the discrete Poisson
     (Yamabe/Cherrier) problem for the interior scale factors that flatten the
     interior, then read off the resulting boundary curvature. Discretely the
     curvatures change as K_new = K + L u, with L the cotangent Laplacian.
  2. Integrate the new exterior angles and boundary lengths into a closed
     boundary curve.
  3. Extend the boundary curve harmonically to the interior (Laplace equation).
"""
import numpy as np
import scipy.sparse.linalg as spla

from .common import (boundary_curve, boundary_loop, cotan_laplacian, edge_lengths,
                     harmonic_extension, vertex_angle_sums)


def flatten(V, F):
    n = len(V)
    l = edge_lengths(V, F)
    L = cotan_laplacian(l, F, n)
    b = boundary_loop(F)
    interior = np.setdiff1d(np.arange(n), b)

    angle_sum = vertex_angle_sums(l, F, n)
    K = 2 * np.pi - angle_sum[interior]   # interior angle defects (Gaussian curvature)
    k = np.pi - angle_sum[b]              # boundary exterior angles (geodesic curvature)

    # 1. Interior scale factors with flat interior (K_new = 0) and u_b = 0 on the boundary,
    #    then the boundary curvature they induce.
    u_b = np.zeros(len(b))
    u_i = spla.spsolve(L[interior][:, interior].tocsc(), -K - L[interior][:, b] @ u_b)
    k_new = k + L[b][:, interior] @ u_i + L[b][:, b] @ u_b

    # 2. Boundary curve with edge lengths scaled by exp((u_i + u_j) / 2).
    lengths = np.linalg.norm(V[np.roll(b, -1)] - V[b], axis=1) * np.exp(0.5 * (u_b + np.roll(u_b, -1)))
    uv_boundary = boundary_curve(lengths, k_new)

    # 3. Harmonic extension to the interior.
    return harmonic_extension(L, b, uv_boundary)
