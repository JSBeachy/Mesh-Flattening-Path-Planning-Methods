"""Correctness tests for the four flattening methods."""
import os
import sys

import numpy as np
import pytest
from scipy.spatial import Delaunay

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flattening import METHODS, arap, bff, cetm, lscm  # noqa: E402
from flattening.common import (boundary_loop, check_disk, corner_angles, cotan_laplacian,  # noqa: E402
                               edge_lengths, vertex_angle_sums)
from evaluation import angle_distortion_deg, log_area_ratio, signed_areas_2d  # noqa: E402


# ----------------------------------------------------------------------------------------
# Test meshes
# ----------------------------------------------------------------------------------------
def disk_mesh(rings=12):
    """Unit disk triangulated from concentric rings (6k points on ring k), CCW faces."""
    pts = [np.zeros((1, 2))]
    for k in range(1, rings + 1):
        a = 2 * np.pi * (np.arange(6 * k) + 0.5 * (k % 2)) / (6 * k)
        pts.append(k / rings * np.stack([np.cos(a), np.sin(a)], axis=1))
    P = np.vstack(pts)
    F = Delaunay(P).simplices.astype(np.int64)
    flip = signed_areas_2d(P, F) < 0
    F[flip] = F[flip][:, ::-1]
    return P, F


def spherical_cap(rings=12, max_angle=np.radians(60)):
    """Disk lifted onto the unit sphere (polar angle proportional to the disk radius)."""
    P, F = disk_mesh(rings)
    r = np.linalg.norm(P, axis=1)
    phi = np.arctan2(P[:, 1], P[:, 0])
    th = r * max_angle
    V = np.stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)], axis=1)
    return V, F


def grid_mesh(nx=16, ny=12, width=1.0, height=0.7):
    """Rectangle triangulated on a regular grid with alternating diagonals, CCW faces."""
    xs, ys = np.meshgrid(np.linspace(0, width, nx + 1), np.linspace(0, height, ny + 1))
    P = np.stack([xs.ravel(), ys.ravel()], axis=1)
    F = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b, c, d = a + 1, a + nx + 2, a + nx + 1
            F += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    return P, np.array(F, dtype=np.int64)


def folded_sheet(fold=np.radians(70)):
    """A rectangle folded along the grid line x = 0.375: an exactly developable polyhedral surface."""
    P, F = grid_mesh()
    V = np.c_[P, np.zeros(len(P))]
    right = V[:, 0] > 0.375 + 1e-12
    x = V[right, 0] - 0.375
    V[right, 0] = 0.375 + x * np.cos(fold)
    V[right, 2] = x * np.sin(fold)
    rng = np.random.default_rng(0)  # random rigid motion
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return V @ Q.T + rng.normal(size=3), F


# ----------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------
def test_test_meshes_are_disks():
    for V, F in [spherical_cap(), folded_sheet()]:
        check_disk(V, F)


def test_lobachevsky_known_values():
    catalan = 0.915965594177219015054603514932
    assert abs(cetm.lobachevsky(np.pi / 4) - catalan / 2) < 1e-14
    assert abs(cetm.clausen2(np.pi / 3) - 1.014941606409653625021202554275) < 1e-14
    for x in [0.0, np.pi / 2, np.pi]:
        assert abs(cetm.lobachevsky(x)) < 1e-14
    # derivative: Lob'(t) = -log(2 sin t)
    t, h = 0.7, 1e-6
    assert abs((cetm.lobachevsky(t + h) - cetm.lobachevsky(t - h)) / (2 * h) + np.log(2 * np.sin(t))) < 1e-8


def test_angle_sums_vary_with_minus_cotan_laplacian():
    """dTheta/du = -L for l_ij = exp((u_i + u_j)/2) l_ij: the relation behind BFF (K_new = K + L u) and CETM."""
    V, F = spherical_cap(6)
    n = len(V)
    l0 = edge_lengths(V, F)
    u = np.random.default_rng(1).normal(scale=0.05, size=n)
    du = np.random.default_rng(2).normal(size=n)

    def theta(u):
        return vertex_angle_sums(l0 * np.exp(0.5 * (u[F[:, [1, 2, 0]]] + u[F[:, [2, 0, 1]]])), F, n)

    h = 1e-6
    fd = (theta(u + h * du) - theta(u - h * du)) / (2 * h)
    L = cotan_laplacian(l0 * np.exp(0.5 * (u[F[:, [1, 2, 0]]] + u[F[:, [2, 0, 1]]])), F, n)
    assert np.allclose(fd, -(L @ du), atol=1e-7)


def test_cetm_energy_gradient():
    V, F = spherical_cap(6)
    n = len(V)
    lam0 = 2 * np.log(edge_lengths(V, F))
    theta_hat = np.full(n, 2 * np.pi)
    u = np.random.default_rng(3).normal(scale=0.05, size=n)
    l = np.exp(0.5 * (lam0 + u[F[:, [1, 2, 0]]] + u[F[:, [2, 0, 1]]]))
    grad = theta_hat - vertex_angle_sums(l, F, n)
    h = 1e-6
    for i in [0, 7, 30, n - 1]:
        e = np.zeros(n)
        e[i] = h
        fd = (cetm.energy(u + e, lam0, F, theta_hat) - cetm.energy(u - e, lam0, F, theta_hat)) / (2 * h)
        assert abs(fd - grad[i]) < 1e-6


# ----------------------------------------------------------------------------------------
# Whole methods
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(METHODS))
def test_developable_surface_is_flattened_isometrically(name):
    V, F = folded_sheet()
    UV = METHODS[name](V, F)
    assert np.all(signed_areas_2d(UV, F) > 0)
    assert np.abs(log_area_ratio(V, F, UV)).max() < 1e-6
    assert angle_distortion_deg(V, F, UV) < 1e-4  # ARAP stops at a finite tolerance


@pytest.mark.parametrize("name", list(METHODS))
def test_curved_surface_gives_valid_map(name):
    V, F = spherical_cap(16, np.radians(75))
    UV = METHODS[name](V, F)
    assert np.all(np.isfinite(UV))
    assert np.all(signed_areas_2d(UV, F) > 0)


def test_lscm_matches_libigl():
    igl = pytest.importorskip("igl")
    V, F = spherical_cap(14, np.radians(80))
    b = boundary_loop(F)
    UV = lscm.flatten(V, F)
    D = np.linalg.norm(V[b][:, None] - V[b][None], axis=2)
    p, r = np.unravel_index(np.argmax(D), D.shape)
    ref, _ = igl.lscm(V, F, np.array([b[p], b[r]]), np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert np.abs(UV - ref).max() < 1e-9


def test_arap_iterations_match_libigl():
    igl = pytest.importorskip("igl")
    V, F = spherical_cap(14, np.radians(80))
    n = len(V)
    L = cotan_laplacian(edge_lengths(V, F), F, n)
    U0 = arap.harmonic_initialization(V, F, L)
    UV = arap.flatten(V, F, tol=0.0, max_iter=25)  # tol = 0 runs exactly max_iter iterations
    data = igl.ARAPData()
    data.energy = igl.ARAP_ENERGY_TYPE_ELEMENTS
    data.with_dynamics = False
    data.max_iter = 25
    igl.arap_precomputation(V, F, 2, np.array([0], dtype=np.int32), data)
    ref = igl.arap_solve(U0[[0]], data, U0)
    assert np.abs(UV - ref).max() < 1e-8 * np.abs(ref).max()


def test_bff_close_to_cetm_on_smooth_surface():
    """BFF (linearized) and CETM (nonlinear) with u_b = 0 are first-order equivalent."""
    V, F = spherical_cap(20, np.radians(45))
    A, B = bff.flatten(V, F), cetm.flatten(V, F)
    la, lb = log_area_ratio(V, F, A), log_area_ratio(V, F, B)
    assert np.abs(la - lb).max() < 0.05 * np.abs(lb).max()


def test_cetm_layout_realizes_flat_metric():
    V, F = spherical_cap(16, np.radians(75))
    n = len(V)
    u = cetm.solve_scale_factors(V, F)
    UV = cetm.flatten(V, F)
    target = edge_lengths(V, F) * np.exp(0.5 * (u[F[:, [1, 2, 0]]] + u[F[:, [2, 0, 1]]]))
    assert np.abs(edge_lengths(UV, F) / target - 1).max() < 1e-8
    b = boundary_loop(F)
    assert np.allclose(u[b], 0)
    interior = np.setdiff1d(np.arange(n), b)
    assert np.abs(vertex_angle_sums(edge_lengths(UV, F), F, n)[interior] - 2 * np.pi).max() < 1e-8


def test_conformal_methods_have_low_angle_distortion():
    V, F = spherical_cap(20, np.radians(70))
    conf = [angle_distortion_deg(V, F, METHODS[m](V, F)) for m in ["LSCM", "BFF", "CETM"]]
    rigid = angle_distortion_deg(V, F, METHODS["ARAP"](V, F))
    assert max(conf) < rigid
    area_conf = [np.abs(log_area_ratio(V, F, METHODS[m](V, F))).mean() for m in ["LSCM", "BFF", "CETM"]]
    area_rigid = np.abs(log_area_ratio(V, F, METHODS["ARAP"](V, F))).mean()
    assert area_rigid < min(area_conf)
