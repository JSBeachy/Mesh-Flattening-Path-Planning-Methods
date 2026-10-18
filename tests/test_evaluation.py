"""Tests for the evaluation measures (area distortion, centers, CBGD, back-projection, spiral, coverage paths)."""
import os
import sys

import numpy as np
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.spatial.transform import Rotation  # noqa: E402

from evaluation import (Geodesics, angle_distortion_deg, back_project, boundary_self_intersections,  # noqa: E402
                        cbgd, coverage_measures, coverage_spiral, length_within_tolerance, locate,
                        log_area_ratio, mapped_center, next_line_distance, pole_of_inaccessibility,
                        principal_axis, raster_passes, ray_distances, spiral_path, subdivision_barycentrics,
                        tool_poses, true_center)
from flattening.common import boundary_loop  # noqa: E402
from test_flattening import disk_mesh, grid_mesh, spherical_cap  # noqa: E402


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


def test_locate_matches_a_scan_of_all_faces_on_a_folded_map():
    P, F = disk_mesh(8)
    UV = P.copy()
    UV[0] = [0.3, 0.05]  # the center vertex pushed past its first ring: its triangles overlap others

    def cross(a, b):
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    A, B, C = UV[F[:, 0]], UV[F[:, 1]], UV[F[:, 2]]  # the scan of all faces that locate() replaces
    area2 = cross(B - A, C - A)
    usable = np.abs(area2) > 1e-12 * np.abs(area2).max()
    area2 = np.where(usable, area2, 1.0)
    rng = np.random.default_rng(5)
    points = np.vstack([rng.uniform(-1.1, 1.1, (400, 2)), UV, 0.5 * (UV[F[:, 0]] + UV[F[:, 1]])])
    faces, weights = locate(UV, F, points)
    for p, f, w in zip(points, faces, weights):
        wa, wb = cross(B - p, C - p) / area2, cross(C - p, A - p) / area2
        W = np.stack([wa, wb, 1.0 - wa - wb], axis=1)
        worst = np.where(usable, W.min(axis=1), -np.inf)
        g = int(np.argmax(worst))
        if worst[g] < -1e-9:
            assert f == -1
        else:
            expected = np.clip(W[g], 0.0, None)
            assert f == g and np.array_equal(w, expected / expected.sum())
    assert np.sum(faces >= 0) > 400


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


def test_boundary_self_intersections():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert boundary_self_intersections(square, np.arange(4)) == 0
    bowtie = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)  # edges 0-1 and 2-3 cross
    assert boundary_self_intersections(bowtie, np.arange(4)) == 1
    a = np.linspace(0, 4 * np.pi, 400, endpoint=False)               # a circle traversed twice
    twice = np.stack([np.cos(a), np.sin(a)], axis=1) * (1 + 0.01 * (a > 2 * np.pi))[:, None]
    assert boundary_self_intersections(twice, np.arange(400)) > 0
    P, F = disk_mesh(8)
    assert boundary_self_intersections(P, boundary_loop(F)) == 0


def test_spiral_path_is_inside_and_evenly_spaced():
    poly = np.array([[0, 0], [100, 0], [100, 60], [0, 60]], dtype=float)
    center = np.array([50.0, 30.0])
    via, length = spiral_path(poly, center, turns=3, step=3.2)
    assert np.allclose(via[0], center)
    assert Path(poly).contains_points(via).all()
    steps = np.linalg.norm(np.diff(via, axis=0), axis=1)
    assert np.all(steps <= 3.2 + 1e-9) and np.median(steps) > 3.1
    assert abs(len(via) - np.ceil(length / 3.2)) <= 1


# ----------------------------------------------------------------------------------------
# Coverage paths
# ----------------------------------------------------------------------------------------
def test_principal_axis_ignores_vertex_spacing():
    P, F = grid_mesh(16, 12, 1.0, 0.7)
    P[:, 0] = P[:, 0] ** 3  # vertices crowd toward x = 0; the region is still the 1 x 0.7 rectangle
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    R = np.array([[c, -s], [s, c]])
    assert np.allclose(principal_axis(P @ R.T, F), [c, s], atol=1e-12)
    assert np.allclose(principal_axis(-P @ R.T, F), [c, s], atol=1e-12)  # sign convention


def test_raster_passes():
    rect = np.array([[0, 0], [1, 0], [1, 0.75], [0, 0.75]], dtype=float)
    passes, lines = raster_passes(rect, np.array([1.0, 0.0]), 0.125)
    assert len(passes) == 6 and np.array_equal(lines, np.arange(6))
    for j, p in enumerate(passes):
        assert np.allclose(p[:, 1], 0.0625 + 0.125 * j)
        assert np.allclose(p[:, 0], [0, 1] if j % 2 == 0 else [1, 0])  # boustrophedon
    # A U shape: lines through both arms give two passes, run in the line's direction.
    u = np.array([[0, 0], [3, 0], [3, 2], [2, 2], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    passes, lines = raster_passes(u, np.array([1.0, 0.0]), 0.5)
    assert np.array_equal(lines, [0, 1, 2, 2, 3, 3])
    assert np.allclose(passes[2][:, 0], [0, 1]) and np.allclose(passes[3][:, 0], [2, 3])
    assert np.allclose(passes[4][:, 0], [3, 2]) and np.allclose(passes[5][:, 0], [1, 0])  # line 3 runs backwards
    assert all(Path(u).contains_points(p.mean(axis=0, keepdims=True))[0] for p in passes)


def test_coverage_spiral_turns_are_at_most_one_spacing_apart():
    a = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    ellipse = np.stack([0.5 + 0.45 * np.cos(a), 0.4 + 0.25 * np.sin(a)], axis=1)
    center = np.array([0.5, 0.4])
    (curve,) = coverage_spiral(ellipse, center, 0.04)
    turns = int(np.ceil(0.98 * 0.45 / 0.04))
    t = np.arange(len(curve)) / 1000  # turn number: 1000 samples per turn
    assert len(curve) == 1000 * (turns + 1) + 1  # spiral turns plus the closing turn
    r = np.linalg.norm(curve - center, axis=1)
    reach = ray_distances(center, ellipse, 2 * np.pi * t)
    spiral, closing = t <= turns, t >= turns
    assert np.allclose(r[spiral], 0.98 * reach[spiral] * t[spiral] / turns)
    assert np.allclose(r[closing], 0.98 * reach[closing])
    assert np.all(0.98 * reach / turns <= 0.04)  # radial distance between consecutive turns


def test_next_line_distance():
    x = np.linspace(0, 1, 11)
    points = np.vstack([np.c_[x, np.zeros(11)], np.c_[x, np.full(11, 1.0)], np.c_[x[:5], np.full(5, 3.0)]])
    line = np.repeat([0, 1, 2], [11, 11, 5])
    d = next_line_distance(points, line)
    assert np.allclose(d[:11], 1.0)
    assert np.allclose(d[11:22], np.hypot(2.0, np.maximum(x - 0.4, 0)))  # the next line ends at x = 0.4
    assert np.all(np.isnan(d[22:]))                                       # the last line has no next one


def test_subdivision_barycentrics():
    for n in [1, 2, 5]:
        w = subdivision_barycentrics(n)
        assert len(w) == n * n and np.all(w > 0) and np.allclose(w.sum(axis=1), 1)
        assert np.allclose(w.mean(axis=0), 1 / 3)
        assert len(np.unique(np.round(w, 12), axis=0)) == n * n
    assert np.allclose(subdivision_barycentrics(1), 1 / 3)


def test_coverage_measures_on_isometric_and_stretched_maps():
    P, F = grid_mesh(20, 14, 1.0, 0.7)
    Q, _ = np.linalg.qr(np.random.default_rng(6).normal(size=(3, 3)))
    V = np.c_[P, np.zeros(len(P))] @ Q.T + 1.0  # the planar grid moved rigidly into 3D
    b = boundary_loop(F)
    for stretch in [1.0, 2.0]:  # the identity map, and a map that doubles u
        UV = P * [stretch, 1.0]
        s = 0.05 * np.sqrt(0.7 * stretch)
        k = 1 / np.sqrt(stretch)  # area-equivalent 3D length per 2D length
        passes, lines = raster_passes(UV[b], principal_axis(UV, F), s)
        assert all(p[0, 1] == p[1, 1] for p in passes)  # the passes run along u, the principal axis
        m, arrays = coverage_measures(V, F, UV, passes, s, lines=lines)
        assert m["unmapped"] == 0 and m["planned_coverage_pct"] > 99.5
        over = arrays["stepover"][np.isfinite(arrays["stepover"])]
        within = length_within_tolerance(V, F, UV)
        if stretch == 1.0:  # isometry: nothing changes
            assert np.abs(over).max() < 1e-9 and m["step_mean_abs_dev_pct"] < 1e-7
            assert abs(m["coverage_pct"] - m["planned_coverage_pct"]) < 1e-9 and within.all()
        else:  # 3D steps along the passes shrink, 3D stepovers grow
            assert np.allclose(over, 1 / k - 1, atol=2e-3)                      # +41 %: gaps between passes
            assert abs(m["step_mean_abs_dev_pct"] - 100 * (1 - 0.5 / k)) < 1e-7  # -29 % along the passes
            assert m["stepover_wide_pct"] == 100 and not within.any()
            assert 84 < m["coverage_pct"] < 88  # bands of width 1.2 k s every s in 3D
        # Passes of 3D length 1 (to within the last sample), against an ideal raster of length A3D / (k s).
        assert abs(m["relative_length"] - len(passes) * k * s / 0.7) < 0.01
