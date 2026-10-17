"""Density-equalizing map (Choi & Rycroft 2018), area-preserving free-boundary flattening.

With the 3D triangle areas as the "population", the density population / (2D
area) is equalized by Gastner-Newman diffusion, so the final map is
area-preserving. Steps, following the paper and the authors' DEM_main code:
  1. Initial map (Alg. 1-2): the boundary is flattened to a convex curve that
     keeps the 3D boundary's edge lengths and distributes its (unsigned)
     curvature, closed by subtracting the linear drift (Eq. 4.10); the interior
     is placed by Tutte's embedding.
  2. Sea (Alg. 4): the map is shrunk into the unit disk, the gap to the unit
     circle is filled with a hexagonal point lattice, and the disk is reflected
     through the circle (z -> 1 / conj(z)), truncated at |z| = 5. The sea gets
     the mean density, so the boundary can move freely.
  3. Iterate (Alg. 5): backward-Euler diffusion of the vertex density with the
     cotangent Laplacian and lumped mass matrix of the current map; move every
     vertex by -dt grad(rho) / rho; stop when std(rho) / mean(rho) < 1e-3 on the
     original vertices (at most 50 steps). dt = min(min/mean, mean/max of the
     density) * (surface area).
  4. Rescale so the 2D area equals the 3D area.
The authors use a constrained Delaunay triangulation for the gap; we use an
ordinary Delaunay triangulation after dropping lattice points inside the
diametral circles of the boundary edges, which makes every boundary edge a
Delaunay edge, and check that the glued mesh is a valid manifold.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.path import Path
from scipy.spatial import Delaunay

from .common import (boundary_loop, cotan_laplacian, edge_lengths, face_areas_from_lengths,
                     harmonic_extension, signed_areas_2d, uniform_laplacian)


def curvature_boundary(P):
    """Alg. 1: closed convex plane polygon with the edge lengths of the 3D boundary polygon P."""
    e = np.roll(P, -1, axis=0) - P
    length = np.linalg.norm(e, axis=1)
    T = e / length[:, None]
    turning = np.arccos(np.clip(np.sum(np.roll(T, 1, axis=0) * T, axis=1), -1.0, 1.0))  # at each vertex
    k = 2 * np.pi * turning / turning.sum()                    # Eq. 4.5: total curvature 2 pi
    theta = np.cumsum(k)                                       # direction of edge i (Eq. 4.7)
    steps = length[:, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    phi = np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])   # Eq. 4.6, phi[-1] closes the loop
    s = np.concatenate([[0.0], np.cumsum(length)])
    phi -= (s / s[-1])[:, None] * (phi[-1] - phi[0])           # Eq. 4.10
    return phi[:-1]


def initial_map(V, F):
    """Alg. 2: Tutte embedding with the curvature-based convex boundary."""
    b = boundary_loop(F)
    return harmonic_extension(uniform_laplacian(F, len(V)), b, curvature_boundary(V[b]))


def construct_sea(uv, F):
    """Alg. 4: returns the vertices and faces of the map glued to its sea (the map's own come first)."""
    n = len(uv)
    b = boundary_loop(F)
    z = uv - uv.mean(axis=0)
    z /= np.linalg.norm(z, axis=1).max()
    h = np.mean(np.linalg.norm(z[b] - np.roll(z[b], -1, axis=0), axis=1))
    z *= 1 - 3 * h
    zb = z[b]

    a = np.arange(0.01, 2 * np.pi, h)
    circle = np.stack([np.cos(a), np.sin(a)], axis=1)
    g = np.arange(-1.1, 1.1 + 1e-12, h)
    X, Y = np.meshgrid(g, g)
    X[1::2, :] += h / 2
    lattice = np.stack([X.ravel(), Y.ravel()], axis=1)
    keep = (Path((1 - h / 2) * circle).contains_points(lattice)
            & ~Path((1 + h / 2) * zb).contains_points(lattice))
    lattice = lattice[keep]
    # Drop lattice points inside a diametral circle of a boundary edge (Gabriel condition).
    mid, rad = 0.5 * (zb + np.roll(zb, -1, axis=0)), 0.5 * np.linalg.norm(np.roll(zb, -1, axis=0) - zb, axis=1)
    near = np.zeros(len(lattice), dtype=bool)
    for s in range(0, len(lattice), 2000):
        q = lattice[s:s + 2000]
        near[s:s + 2000] = np.any(np.linalg.norm(q[:, None] - mid[None], axis=2) <= rad[None] * (1 + 1e-9), axis=1)
    lattice = lattice[~near]

    # Delaunay triangulation of boundary + lattice + circle; keep triangles outside the map.
    pts = np.vstack([zb, lattice, circle])
    tri = Delaunay(pts).simplices
    centroid = pts[tri].mean(axis=1)
    tri = tri[~Path(zb).contains_points(centroid)]
    ids = np.concatenate([b, n + np.arange(len(lattice) + len(circle))])
    gap = ids[tri]
    verts = np.vstack([z, lattice, circle])
    flip = signed_areas_2d(verts, gap) < 0
    gap[flip] = gap[flip][:, ::-1]
    disk = np.vstack([F, gap])

    # Reflect the disk through the unit circle; circle vertices are shared, faces reversed.
    c = verts[disk].mean(axis=1)
    reflect = disk[np.linalg.norm(c, axis=1) >= 0.2]                 # |1 / conj(c)| <= 5
    on_circle = np.zeros(len(verts), dtype=bool)
    on_circle[len(verts) - len(circle):] = True
    moved = np.setdiff1d(np.unique(reflect), np.flatnonzero(on_circle))
    new_id = np.arange(len(verts))
    new_id[moved] = len(verts) + np.arange(len(moved))
    zr = verts[moved] / np.sum(verts[moved] ** 2, axis=1, keepdims=True)  # 1 / conj(z)
    faces = np.vstack([disk, new_id[reflect][:, ::-1]])
    verts = np.vstack([verts, zr])

    # Sanity checks: consistently oriented manifold in which the map's boundary edges are interior.
    he = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keys = set(map(tuple, he.tolist()))
    twins = all((int(j), int(i)) in keys for i, j in zip(b, np.roll(b, -1)))
    if len(keys) != len(he) or not twins or np.any(signed_areas_2d(verts, faces) <= 0):
        raise RuntimeError("DEM: invalid sea triangulation")
    return verts, faces


def lumped_mass(uv, F, n):
    return sp.diags(np.bincount(F.ravel(), np.repeat(face_areas_from_lengths(edge_lengths(uv, F)) / 3, 3),
                                minlength=n))


def face_to_vertex(uv, F, n):
    """Area-weighted face-to-vertex averaging matrix (n, m)."""
    area = face_areas_from_lengths(edge_lengths(uv, F))
    M = sp.csr_matrix((np.repeat(area, 3), (F.ravel(), np.repeat(np.arange(len(F)), 3))), shape=(n, len(F)))
    return sp.diags(1.0 / np.asarray(M.sum(axis=1)).ravel()) @ M


def face_gradients(uv, F, g):
    """(m, 2) gradient of the piecewise-linear function g on each triangle of the plane mesh uv."""
    p0, p1, p2 = uv[F[:, 0]], uv[F[:, 1]], uv[F[:, 2]]
    area2 = signed_areas_2d(uv, F)[:, None] * 2
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])     # 90 degree rotation: N x e for a CCW triangle
    return ((g[F[:, 0], None] * (p2 - p1) + g[F[:, 1], None] * (p0 - p2) + g[F[:, 2], None] * (p1 - p0))
            @ rot.T) / area2


def density_equalizing_map(V, F, population, uv0, epsilon=1e-3, max_steps=50, stats=None):
    """Alg. 5 (the authors' DEM_main): deform the initial map uv0 until the density is equalized.

    If a dict is passed as stats, the number of diffusion steps is stored in stats["steps"].
    """
    n = len(V)
    area = face_areas_from_lengths(edge_lengths(V, F)).sum()
    population = population * (area / population.sum())
    r, faces = construct_sea(uv0, F)
    r *= np.sqrt(area / face_areas_from_lengths(edge_lengths(r, F)).sum())
    N = len(r)

    rho_f = population / face_areas_from_lengths(edge_lengths(r, F))
    rho = np.empty(N)
    rho[:n] = face_to_vertex(r[:n], F, n) @ rho_f
    rho[n:] = rho[:n].mean()
    dt = min(rho[:n].min() / rho[:n].mean(), rho[:n].mean() / rho[:n].max()) * area

    step = 1  # counted as in DEM_main, which stops after max_steps - 1 updates
    while np.std(rho[:n], ddof=1) / np.mean(rho[:n]) > epsilon:
        L = cotan_laplacian(edge_lengths(r, faces), faces, N)
        A = lumped_mass(r, faces, N)
        rho_new = spla.spsolve((A + dt * L).tocsc(), A @ rho)
        if not np.all(np.isfinite(rho_new)):
            break
        grad = face_to_vertex(r, faces, N) @ face_gradients(r, faces, rho_new)
        r = r - dt * grad / rho_new[:, None]
        rho = rho_new
        step += 1
        if step >= max_steps:
            break

    if stats is not None:
        stats["steps"] = step - 1
    uv = r[:n]
    return uv * np.sqrt(area / face_areas_from_lengths(edge_lengths(uv, F)).sum())


def flatten(V, F):
    """Area-preserving map: the population of each triangle is its 3D area."""
    return density_equalizing_map(V, F, face_areas_from_lengths(edge_lengths(V, F)), initial_map(V, F))
