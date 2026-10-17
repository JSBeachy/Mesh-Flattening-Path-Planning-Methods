"""Tests for the evaluation measures (area distortion, centers, CBGD, back-projection, spiral)."""
import os
import sys

import numpy as np
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.spatial.transform import Rotation  # noqa: E402

from evaluation import (Geodesics, angle_distortion_deg, back_project, cbgd, locate,  # noqa: E402
                        log_area_ratio, mapped_center, pole_of_inaccessibility, spiral_path,
                        tool_poses, true_center)
from flattening.common import boundary_loop  # noqa: E402
from test_flattening import disk_mesh, spherical_cap  # noqa: E402


def test_pole_of_inaccessibility_square_and_circle():
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    assert np.allclose(pole_of_inaccessibility(square), [1, 1], atol=1e-3)
    a = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    circle = np.stack([3 + np.cos(a), -1 + np.sin(a)], axis=1)
    assert np.allclose(pole_of_inaccessibility(circle), [3, -1], atol=1e-3)


def test_pole_of_inaccessibility_l_shape():
    # Largest inscribed circle touches x = 0, y = 0 and the reflex corner (1, 1).
    poly = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    c = np.sqrt(2) / (1 + np.sqrt(2))
    assert np.allclose(pole_of_inaccessibility(poly, rel_tol=1e-6), [c, c], atol=1e-3)


def test_locate_and_back_project():
    P, F = disk_mesh(8)
    V = np.c_[P, P[:, 0] ** 2]  # a 3D lift of the planar mesh
    centroids = P[F].mean(axis=1)
    faces, w = locate(P, F, centroids)
    assert np.array_equal(faces, np.arange(len(F)))
    assert np.allclose(w, 1 / 3)
    assert np.allclose(back_project(V, F, faces, w), V[F].mean(axis=1))
    faces, w = locate(P, F, P)  # vertices lie on several faces: any of them, with weight 1
    assert np.all(faces >= 0)
    assert np.allclose(back_project(V, F, faces, w), V)
    faces, _ = locate(P, F, np.array([[2.0, 2.0]]))
    assert faces[0] == -1


def test_identity_map_has_no_distortion():
    P, F = disk_mesh(8)
    V = np.c_[P, np.zeros(len(P))]
    assert np.abs(log_area_ratio(V, F, P)).max() < 1e-12
    assert angle_distortion_deg(V, F, P) < 1e-9


def test_centers_and_cbgd_on_symmetric_cap():
    V, F = spherical_cap(10, np.radians(60))
    b = boundary_loop(F)
    geo = Geodesics(V, F)
    p_t = true_center(geo, b)
    assert p_t == 0  # the pole of the cap
    UV = V[:, :2] / np.linalg.norm(V[:, :2], axis=1, keepdims=True).clip(1e-12) * np.arccos(V[:, 2:3])
    center_2d, point_3d, p_m = mapped_center(V, F, UV, b)
    assert np.allclose(center_2d, 0, atol=1e-3) and p_m == p_t
    delta, d_true, d_mapped = cbgd(geo, b, p_t, p_m)
    assert np.all(delta == 0)
    assert np.allclose(d_true, np.radians(60), rtol=1e-2)  # geodesic radius of the cap


def test_tool_poses_are_proper_rotations_facing_the_surface():
    V, F = spherical_cap(10, np.radians(60))
    rng = np.random.default_rng(4)
    faces = rng.integers(0, len(F), 20)
    w = rng.dirichlet(np.ones(3), 20)
    P = back_project(V, F, faces, w)
    positions, quats = tool_poses(V, F, faces, P, offset=0.1)
    R = Rotation.from_quat(quats).as_matrix()
    n = np.cross(V[F[faces, 1]] - V[F[faces, 0]], V[F[faces, 2]] - V[F[faces, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    assert np.allclose(np.linalg.det(R), 1)
    assert np.allclose(R[:, :, 2], -n)                        # tool z axis along -N
    assert np.allclose(np.sum(R[:, :, 0] * n, axis=1), 0)     # x axis tangent to the surface
    d = P[1] - P[0]
    assert np.dot(R[0, :, 0], d - np.dot(d, n[0]) * n[0]) > 0  # x axis points toward the next via point
    assert np.allclose(positions, P + 0.1 * n)


def test_spiral_path_is_inside_and_evenly_spaced():
    poly = np.array([[0, 0], [100, 0], [100, 60], [0, 60]], dtype=float)
    center = np.array([50.0, 30.0])
    via, length = spiral_path(poly, center, turns=3, step=3.2)
    assert np.allclose(via[0], center)
    assert Path(poly).contains_points(via).all()
    steps = np.linalg.norm(np.diff(via, axis=0), axis=1)
    assert np.all(steps <= 3.2 + 1e-9) and np.median(steps) > 3.1
    assert abs(len(via) - np.ceil(length / 3.2)) <= 1
