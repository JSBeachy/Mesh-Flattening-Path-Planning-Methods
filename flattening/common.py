"""Mesh utilities shared by the flattening methods.

Conventions used throughout:
  * V is a (n, 3) array of vertex positions, F a (m, 3) array of vertex indices.
  * Per-face quantities indexed by a corner c in {0, 1, 2} refer to the edge
    *opposite* that corner, i.e. the edge joining F[:, (c+1) % 3] and F[:, (c+2) % 3].
  * The cotangent Laplacian is positive semi-definite:
    L_ij = -(cot a_ij + cot b_ij) / 2 for neighbors, L_ii = -sum_j L_ij.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh
from scipy.sparse.csgraph import connected_components


def load_mesh(path):
    """Load a triangle mesh, clean it and check that it is a topological disk."""
    mesh = trimesh.load(path, process=True, force="mesh")
    V, F = clean_mesh(mesh.vertices, mesh.faces)
    check_disk(V, F)
    return V, F


def clean_mesh(V, F):
    """Mesh cleaning as in Sec. 3 of the paper.

    Merges duplicate vertices, removes degenerate, duplicate and unreferenced
    faces, makes the face winding consistent and keeps the largest connected
    component.
    """
    mesh = trimesh.Trimesh(np.asarray(V, dtype=float), np.asarray(F, dtype=np.int64), process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(mesh)
    adj = mesh.face_adjacency
    m = len(mesh.faces)
    _, label = connected_components(sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(m, m)),
                                    directed=False)
    mesh.update_faces(label == np.bincount(label).argmax())
    mesh.remove_unreferenced_vertices()
    return np.array(mesh.vertices, dtype=float), np.array(mesh.faces, dtype=np.int64)


def check_disk(V, F):
    """Raise ValueError unless (V, F) is an oriented manifold disk (one boundary loop, chi = 1)."""
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    if len(np.unique(he, axis=0)) != len(he):
        raise ValueError("inconsistent orientation or non-manifold edge")
    n_edges = len(np.unique(np.sort(he, axis=1), axis=0))
    if len(np.unique(F)) != len(V):
        raise ValueError("unreferenced vertices")
    if len(V) - n_edges + len(F) != 1:
        raise ValueError("Euler characteristic is not 1 (not a disk)")
    boundary_loop(F)  # raises if the boundary is not a single simple loop


def boundary_loop(F):
    """Ordered boundary vertices, oriented consistently with the faces (CCW for an upward normal)."""
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    n = int(F.max()) + 1
    keys = he[:, 0] * n + he[:, 1]
    twins = he[:, 1] * n + he[:, 0]
    bhe = he[~np.isin(twins, keys)]  # half-edges without an opposite half-edge
    if len(bhe) == 0:
        raise ValueError("mesh has no boundary")
    nxt = dict(zip(bhe[:, 0].tolist(), bhe[:, 1].tolist()))
    if len(nxt) != len(bhe):
        raise ValueError("non-manifold boundary vertex")
    loop = [int(bhe[0, 0])]
    while True:
        v = nxt[loop[-1]]
        if v == loop[0]:
            break
        loop.append(v)
        if len(loop) > len(bhe):
            raise ValueError("boundary is not a simple loop")
    if len(loop) != len(bhe):
        raise ValueError("mesh has more than one boundary loop")
    return np.array(loop)


def edge_lengths(V, F):
    """(m, 3) array; column c holds the length of the edge opposite corner c."""
    return np.stack([np.linalg.norm(V[F[:, (c + 1) % 3]] - V[F[:, (c + 2) % 3]], axis=1)
                     for c in range(3)], axis=1)


def corner_angles(l):
    """Interior angles from edge lengths (law of cosines).

    Triangles violating the triangle inequality get the degenerate angles
    (pi, 0, 0), which is the extension used by Springborn et al. (2008).
    """
    l1, l2 = np.roll(l, -1, axis=1), np.roll(l, -2, axis=1)
    return np.arccos(np.clip((l1 ** 2 + l2 ** 2 - l ** 2) / (2 * l1 * l2), -1.0, 1.0))


def face_areas_from_lengths(l):
    """Triangle areas from edge lengths (Heron); 0 for degenerate triangles."""
    a, b, c = l[:, 0], l[:, 1], l[:, 2]
    s = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)
    return 0.25 * np.sqrt(np.maximum(s, 0.0))


def cotangents(l):
    """(m, 3) cotangents of the corner angles; 0 for degenerate triangles."""
    l1, l2 = np.roll(l, -1, axis=1), np.roll(l, -2, axis=1)
    area = face_areas_from_lengths(l)
    ok = area > 1e-12 * np.max(l, axis=1) ** 2
    cot = np.zeros_like(l)
    cot[ok] = (l1[ok] ** 2 + l2[ok] ** 2 - l[ok] ** 2) / (4 * area[ok, None])
    return cot


def cotan_laplacian(l, F, n):
    """Positive semi-definite cotangent Laplacian built from per-face edge lengths."""
    w = 0.5 * cotangents(l)
    i, j = F[:, [1, 2, 0]].ravel(), F[:, [2, 0, 1]].ravel()  # edge opposite each corner
    w = w.ravel()
    rows = np.concatenate([i, j, i, j])
    cols = np.concatenate([j, i, i, j])
    vals = np.concatenate([-w, -w, w, w])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def vertex_angle_sums(l, F, n):
    """Sum of corner angles around each vertex."""
    return np.bincount(F.ravel(), corner_angles(l).ravel(), minlength=n)


def uniform_laplacian(F, n):
    """Graph Laplacian with unit edge weights (Tutte's barycentric embedding)."""
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    i, j, w = E[:, 0], E[:, 1], np.ones(len(E))
    return sp.csr_matrix((np.concatenate([-w, -w, w, w]),
                          (np.concatenate([i, j, i, j]), np.concatenate([j, i, i, j]))), shape=(n, n))


def local_frames(V, F):
    """(m, 3, 2) isometric 2D coordinates of each triangle's corners (corner 0 at the origin, edge 0-1 along x)."""
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    l1 = np.linalg.norm(e1, axis=1)
    X = np.zeros((len(F), 3, 2))
    X[:, 1, 0] = l1
    X[:, 2, 0] = np.einsum("ij,ij->i", e2, e1) / l1
    X[:, 2, 1] = np.linalg.norm(np.cross(e1, e2), axis=1) / l1
    return X


def gradient_operators(V, F):
    """Sparse (m, n) matrices Dx, Dy giving the gradient of a piecewise-linear function in each
    triangle's local frame. The Jacobian of a map UV is J = [[Dx u, Dy u], [Dx v, Dy v]]."""
    X = local_frames(V, F)
    a, b, c = X[:, 1, 0], X[:, 2, 0], X[:, 2, 1]  # edge matrix [x1 - x0, x2 - x0] = [[a, b], [0, c]]
    rows = np.repeat(np.arange(len(F)), 3)
    dx = np.stack([-1 / a, 1 / a, np.zeros_like(a)], axis=1)
    dy = np.stack([b / (a * c) - 1 / c, -b / (a * c), 1 / c], axis=1)
    shape = (len(F), len(V))
    return (sp.csr_matrix((dx.ravel(), (rows, F.ravel())), shape=shape),
            sp.csr_matrix((dy.ravel(), (rows, F.ravel())), shape=shape))


def signed_areas_2d(UV, F):
    a, b = UV[F[:, 1]] - UV[F[:, 0]], UV[F[:, 2]] - UV[F[:, 0]]
    return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def orient_ccw(UV, F):
    """Mirror a map whose total signed area is negative (some methods fix the map only up to a reflection)."""
    return UV * np.array([-1.0, 1.0]) if signed_areas_2d(UV, F).sum() < 0 else UV


def circle_boundary(V, b):
    """Boundary vertices on a circle with the same circumference as the boundary, spaced by arc length."""
    seg = np.linalg.norm(V[np.roll(b, -1)] - V[b], axis=1)
    t = 2 * np.pi * np.concatenate([[0.0], np.cumsum(seg)[:-1]]) / seg.sum()
    return seg.sum() / (2 * np.pi) * np.stack([np.cos(t), np.sin(t)], axis=1)


def harmonic_extension(L, b, xb):
    """Solve the Dirichlet problem L_II x_I = -L_IB x_B; returns x for all vertices.

    L may be any (possibly non-symmetric) Laplacian-like matrix, e.g. with
    uniform (Tutte), mean-value (Floater) or cotangent weights.
    """
    n = L.shape[0]
    interior = np.setdiff1d(np.arange(n), b)
    x = np.zeros((n, xb.shape[1]))
    x[b] = xb
    L = L.tocsr()
    lu = spla.splu(L[interior][:, interior].tocsc())
    x[interior] = lu.solve(-(L[interior][:, b] @ xb))
    return x


def boundary_curve(lengths, turning_angles):
    """Closed planar polygon from target edge lengths and exterior (turning) angles.

    Edge p runs from boundary vertex p to p+1 and the curve turns by
    turning_angles[p] at vertex p. The lengths are changed as little as
    possible (in the 1/length-weighted least-squares sense) so that the curve
    closes exactly, as in Sawhney & Crane (2017), Sec. 4.3.
    """
    phi = np.cumsum(turning_angles)
    T = np.stack([np.cos(phi), np.sin(phi)])  # 2 x nb unit tangents
    mu = np.linalg.solve((T * lengths) @ T.T, T @ lengths)
    closed = lengths - lengths * (T.T @ mu)
    steps = closed[:, None] * T.T
    return np.vstack([np.zeros(2), np.cumsum(steps, axis=0)[:-1]])
