"""Build the eight test meshes in meshes/ (stand-ins for the paper's objects, see README).

Three meshes come from libigl tutorial models (downloaded from a pinned commit),
five are procedural. Every mesh is cleaned and isotropically remeshed with
PyMeshLab using the paper's settings (10 iterations, 30 degree crease angle),
with the target edge length chosen to give about the face count of the
corresponding object in the paper.

Usage:  python make_meshes.py
"""
import os
import urllib.request

import numpy as np
import pymeshlab
import trimesh
from scipy.spatial import Delaunay, cKDTree

from flattening.common import boundary_loop, check_disk, clean_mesh, corner_angles, edge_lengths

ROOT = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(ROOT, "meshes")
DOWNLOAD_DIR = os.path.join(MESH_DIR, "downloads")
LIBIGL_DATA = ("https://raw.githubusercontent.com/libigl/libigl-tutorial-data/"
               "644dd4104843b6d736745d9dafbd70bf8d175648/")


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
def download(name):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    path = os.path.join(DOWNLOAD_DIR, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(LIBIGL_DATA + name, path)
    mesh = trimesh.load(path, process=True, force="mesh")
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)


def slice_mesh(V, F, planes):
    """Keep the part of the mesh on the positive side of each (origin, normal) plane."""
    for origin, normal in planes:
        V, F, _ = trimesh.intersections.slice_faces_plane(V, F, np.asarray(normal, float),
                                                         np.asarray(origin, float))
    return V, F


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def orient_ccw(P, F):
    e1, e2 = P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]
    flip = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] < 0
    F = F.copy()
    F[flip] = F[flip][:, ::-1]
    return F


def star_domain(radius_fn, h=0.01):
    """Isotropic triangulation (spacing ~h) of the star-shaped domain r <= radius_fn(angle).

    Boundary points are spaced uniformly along the outline, interior points lie
    on a hexagonal lattice at least 0.6 h from the outline. Returns the planar
    points, faces, the normalized radius rho = r / radius_fn(angle) and the angle.
    """
    t = np.linspace(-np.pi, np.pi, 40001)[:-1]
    outline = radius_fn(t)[:, None] * np.stack([np.cos(t), np.sin(t)], axis=1)
    closed = np.vstack([outline, outline[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    q = np.linspace(0.0, s[-1], int(round(s[-1] / h)), endpoint=False)
    boundary = np.stack([np.interp(q, s, closed[:, 0]), np.interp(q, s, closed[:, 1])], axis=1)

    r_max = np.abs(outline).max()
    xs, ys = np.arange(-r_max, r_max, h), np.arange(-r_max, r_max, h * np.sqrt(3) / 2)
    X, Y = np.meshgrid(xs, ys)
    X = X + 0.5 * h * (np.arange(len(ys)) % 2)[:, None]
    lattice = np.stack([X.ravel(), Y.ravel()], axis=1)
    inside = np.hypot(*lattice.T) < radius_fn(np.arctan2(lattice[:, 1], lattice[:, 0]))
    far = cKDTree(outline).query(lattice)[0] > 0.6 * h
    P = np.vstack([boundary, lattice[inside & far]])

    F = Delaunay(P).simplices.astype(np.int64)
    c = P[F].mean(axis=1)
    F = orient_ccw(P, F[np.hypot(*c.T) < radius_fn(np.arctan2(c[:, 1], c[:, 0]))])
    phi = np.arctan2(P[:, 1], P[:, 0])
    return P, F, np.hypot(*P.T) / radius_fn(phi), phi


def grid(xs, ys):
    """Rectangle triangulated on a regular grid with alternating diagonals."""
    X, Y = np.meshgrid(xs, ys)
    nx, ny = len(xs) - 1, len(ys) - 1
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    a = (j * (nx + 1) + i).ravel()
    b, c, d = a + 1, a + nx + 2, a + nx + 1
    even = ((i + j) % 2 == 0).ravel()
    F = np.concatenate([np.stack([a, b, c], 1)[even], np.stack([a, c, d], 1)[even],
                        np.stack([a, b, d], 1)[~even], np.stack([b, c, d], 1)[~even]])
    return np.stack([X.ravel(), Y.ravel()], axis=1), F


def subdivided_triangle(A, B, C, n):
    """Triangle ABC split into n^2 congruent triangles (same orientation as ABC)."""
    idx, pts = {}, []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            idx[i, j] = len(pts)
            pts.append(A + (B - A) * (i / n) + (C - A) * (j / n))
    F = []
    for i in range(n):
        for j in range(n - i):
            F.append([idx[i, j], idx[i + 1, j], idx[i, j + 1]])
            if i + j < n - 1:
                F.append([idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]])
    return np.array(pts), np.array(F)


def remesh(V, F, target_faces):
    """PyMeshLab isotropic explicit remeshing (10 iterations, 30 degree crease angle) to ~target_faces."""
    h = np.sqrt(4 * trimesh.Trimesh(V, F, process=False).area / (np.sqrt(3) * target_faces))
    for _ in range(6):
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(V, F))
        ms.meshing_isotropic_explicit_remeshing(iterations=10, targetlen=pymeshlab.PureValue(h), featuredeg=30)
        out = ms.current_mesh()
        ratio = out.face_number() / target_faces
        if abs(ratio - 1) < 0.02:
            break
        h *= np.sqrt(ratio)
    return out.vertex_matrix(), out.face_matrix()


# ----------------------------------------------------------------------------------------
# Test objects
# ----------------------------------------------------------------------------------------
def wash_basin():
    """Rectangular slab with a rounded-rectangle bowl and a small drain recess."""
    P, F = grid(np.linspace(-1, 1, 361), np.linspace(-0.55, 0.55, 199))
    x, y = P.T
    s = ((np.abs(x) / 0.82) ** 4 + (np.abs(y) / 0.40) ** 4) ** 0.25  # superellipse "radius" of the bowl
    t = smoothstep((1 - s) / 0.3)                                     # 0 on the flat rim, 1 on the floor
    z = -t * (0.25 + 0.05 * (1 - s ** 2)) - 0.03 * np.exp(-(x ** 2 + y ** 2) / 0.05 ** 2)
    return np.c_[x, y, z], F


def turtle():
    """Domed, bumpy shell with head, four flippers and a tail."""
    lobes = [(90, 0.45, 0.20), (35, 0.65, 0.19), (145, 0.65, 0.19),
             (225, 0.45, 0.17), (315, 0.45, 0.17), (270, 0.22, 0.07)]  # (angle deg, length, width rad)

    def radius(phi):  # super-Gaussian lobes give limbs with steep sides and blunt ends
        r = np.ones_like(phi)
        for angle, length, width in lobes:
            r += length * np.exp(-(np.angle(np.exp(1j * (phi - np.radians(angle)))) / width) ** 4)
        return r

    P, F, rho, _ = star_domain(radius)
    x, y = P.T
    rim = np.clip(1 - (np.hypot(x, y) / 0.95) ** 2, 0, None)
    centers = [(0, 0)] + [(0.5 * np.cos(a), 0.5 * np.sin(a)) for a in np.radians(np.arange(30, 360, 60))]
    scutes = 0.02 * rim * sum(np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 0.02) for cx, cy in centers)
    z = 0.45 * rim ** 2 + scutes + 0.10 * (1 - rho ** 2)
    return np.c_[x, y, z], F


def face_mask():
    """Front of a human head scan (libigl face.obj): in front of the ears, above the chin."""
    V, F = download("face.obj")
    return slice_mesh(V, F, [((0, 0, 0.0), (0, 0, 1)), ((0, 156.0, 0), (0, 1, 0))])


def bump_cap():
    """Near-hemispherical ellipsoidal dome with three ribs and a front visor (creased junction)."""
    P, F, _, phi = star_domain(lambda a: 1 + 0.55 * np.clip(np.cos(a), 0, None) ** 1.5)
    r = np.hypot(*P.T)
    th = np.radians(80) * np.minimum(r, 1)
    ribs = sum(np.exp(-((P[:, 1] - y0) / 0.06) ** 2) for y0 in (-0.35, 0.0, 0.35))
    scale = 1 + 0.03 * ribs * np.clip(1 - r ** 2, 0, None)
    dome = scale[:, None] * np.stack([1.05 * np.sin(th) * np.cos(phi), 0.90 * np.sin(th) * np.sin(phi),
                                      0.80 * np.cos(th)], axis=1)
    visor = np.clip(r - 1, 0, None)[:, None] * np.stack([np.cos(phi), np.sin(phi), np.full_like(phi, -0.15)], 1)
    return dome + visor, F


def animal_head():
    """Top of the libigl camel head (ears, eyes, forehead, upper snout); stands in for the kitten face."""
    V, F = download("camelhead.off")
    return slice_mesh(V, F, [((0, -0.05, 0), (0, 1, 0))])


def lilium_top():
    """The libigl lilium model, which matches the paper's 'Lilium top'."""
    return download("lilium.obj")


def four_sided_star():
    """Star pyramid: eight planar facets meeting along creases at the apex."""
    apex = np.array([0.0, 0.0, 0.45])
    corners = [np.array([np.cos(a), np.sin(a), 0.0]) * (1.0 if k % 2 == 0 else 0.35)
               for k, a in enumerate(np.radians(np.arange(0, 360, 45)))]
    Vs, Fs, offset = [], [], 0
    for k in range(8):
        Vt, Ft = subdivided_triangle(apex, corners[k], corners[(k + 1) % 8], 60)
        Vs.append(Vt)
        Fs.append(Ft + offset)
        offset += len(Vt)
    return np.vstack(Vs), np.vstack(Fs)


def car_body():
    """Rounded-rectangle body shell with steep sides and a raised cabin."""
    a, b, p = 1.0, 0.45, 6.0
    P, F, rho, _ = star_domain(lambda t: (np.abs(np.cos(t) / a) ** p + np.abs(np.sin(t) / b) ** p) ** (-1 / p))
    x, y = P.T
    sides = (1 - np.exp(-(1 - rho) / 0.04)) / (1 - np.exp(-1 / 0.04))
    body = 0.22 * sides * (1 - 0.25 * x ** 2)
    s_cabin = ((np.abs(x + 0.12) / 0.48) ** 4 + (np.abs(y) / 0.36) ** 4) ** 0.25
    cabin = 0.20 * smoothstep((1 - s_cabin) / 0.4)
    return np.c_[x, y, body + cabin], F


# (file name, builder, face count of the corresponding object in the paper)
MESHES = [
    ("wash_basin", wash_basin, 10902),
    ("turtle", turtle, 7600),
    ("face_mask", face_mask, 11168),
    ("bump_cap", bump_cap, 15082),
    ("animal_head", animal_head, 31954),
    ("lilium_top", lilium_top, 1844),
    ("four_sided_star", four_sided_star, 5344),
    ("car_body", car_body, 12763),
]


def main():
    os.makedirs(MESH_DIR, exist_ok=True)
    for name, build, target_faces in MESHES:
        V, F = clean_mesh(*build())
        V, F = clean_mesh(*remesh(V, F, target_faces))
        check_disk(V, F)
        trimesh.Trimesh(V, F, process=False).export(os.path.join(MESH_DIR, name + ".obj"))
        angles = np.degrees(corner_angles(edge_lengths(V, F)))
        print(f"{name:16s} V={len(V):6d} F={len(F):6d} boundary={len(boundary_loop(F)):4d} "
              f"angles=[{angles.min():.1f}, {angles.max():.1f}] deg")


if __name__ == "__main__":
    main()
