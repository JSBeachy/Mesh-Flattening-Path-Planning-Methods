"""Tests for the methods added by the extension: SCP, ABF++, SLIM, MDS, DEM and Floater."""
import os
import sys

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import face_areas_3d, log_area_ratio, signed_areas_2d  # noqa: E402
from flattening import METHODS, abf, dem, floater, lscm, mds, scp, slim  # noqa: E402
from flattening.common import (boundary_loop, check_disk, corner_angles, edge_lengths,  # noqa: E402
                               face_areas_from_lengths, gradient_operators, harmonic_extension)
from test_flattening import disk_mesh, grid_mesh, spherical_cap  # noqa: E402


def similarity_residual(A, B):
    """Relative residual of the best similarity (rotation, scale, translation) mapping A onto B."""
    a, b = A - A.mean(axis=0), B - B.mean(axis=0)
    za, zb = a[:, 0] + 1j * a[:, 1], b[:, 0] + 1j * b[:, 1]
    return np.linalg.norm(np.vdot(za, zb) / np.vdot(za, za) * za - zb) / np.linalg.norm(zb)


def orthogonal_residual(A, B):
    """Relative residual of the best orthogonal map (rotation or reflection) of A onto B."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return np.linalg.norm(A @ U @ Vt - B) / np.linalg.norm(B)


# ----------------------------------------------------------------------------------------
# SCP
# ----------------------------------------------------------------------------------------
def scp_operators(V, F):
    Q, b = lscm.conformal_matrix(V, F)
    on_boundary = np.zeros(2 * len(V))
    on_boundary[b] = on_boundary[b + len(V)] = 1.0
    return Q, b, on_boundary


def stacked(UV):
    return np.concatenate([UV[:, 0], UV[:, 1]])


def test_scp_is_the_generalized_eigenvector():
    """L_C u = lambda B u (Mullen et al., Eq. 5) for the boundary-centered map."""
    V, F = spherical_cap(12, np.radians(80))
    Q, b, B = scp_operators(V, F)
    UV = scp.flatten(V, F)
    x = stacked(UV - UV[b].mean(axis=0))
    lam = (x @ (Q @ x)) / (x @ (B * x))
    assert np.linalg.norm(Q @ x - lam * B * x) < 1e-8 * np.linalg.norm(lam * B * x)


def test_scp_matches_a_dense_solve_of_the_papers_formulation():
    """Reference: Eq. 7 with eps I, solved densely with a tiny eps (its eigenvector tends to Eq. 5's as eps -> 0)."""
    V, F = spherical_cap(6, np.radians(70))
    n = len(V)
    Q, b, B = scp_operators(V, F)
    e = np.zeros((2 * n, 2))
    e[b, 0] = e[b + n, 1] = 1.0
    _, vecs = sla.eigh(np.diag(B) - e @ e.T / len(b), Q.toarray() + 1e-13 * np.eye(2 * n))
    ref = np.stack([vecs[:n, -1], vecs[n:, -1]], axis=1)
    assert similarity_residual(scp.flatten(V, F), ref) < 1e-8


def test_scp_minimizes_the_quotient_that_lscm_only_bounds():
    """SCP minimizes E_C / (boundary moment of inertia); LSCM's two pins cannot do better."""
    V, F = spherical_cap(14, np.radians(80))
    Q, b, B = scp_operators(V, F)

    def quotient(UV):
        x = stacked(UV - UV[b].mean(axis=0))
        return (x @ (Q @ x)) / (x @ (B * x))

    assert quotient(scp.flatten(V, F)) < quotient(METHODS["LSCM"](V, F))
    assert signed_areas_2d(scp.flatten(V, F), F).min() > 0


# ----------------------------------------------------------------------------------------
# ABF++
# ----------------------------------------------------------------------------------------
def test_abf_angles_satisfy_the_constraints_and_are_realized_exactly():
    V, F = spherical_cap(14, np.radians(80))
    n = len(V)
    alpha = abf.solve_angles(V, F)
    interior = np.setdiff1d(np.arange(n), boundary_loop(F))
    assert np.abs(alpha.sum(axis=1) - np.pi).max() < 1e-9
    assert np.abs(np.bincount(F.ravel(), alpha.ravel(), minlength=n)[interior] - 2 * np.pi).max() < 1e-9
    log_sin = np.log(np.sin(alpha))
    nxt = np.bincount(F.ravel(), np.roll(log_sin, -1, axis=1).ravel(), minlength=n)
    prv = np.bincount(F.ravel(), np.roll(log_sin, 1, axis=1).ravel(), minlength=n)
    assert np.abs(nxt - prv)[interior].max() < 1e-9   # wheel condition: consistent edge lengths
    UV = abf.angles_to_uv(V, F, alpha)
    assert np.abs(corner_angles(edge_lengths(UV, F)) - alpha).max() < 1e-8


def test_abf_angles_minimize_the_abf_energy_over_flat_maps():
    V, F = spherical_cap(14, np.radians(80))
    beta, _ = abf.target_angles(V, F)

    def energy(alpha):
        return np.sum((alpha - beta) ** 2 / beta ** 2)

    best = energy(abf.solve_angles(V, F))
    for m in ["LSCM", "BFF", "CETM", "SCP", "ARAP"]:  # their angles are feasible ABF solutions
        assert best < energy(corner_angles(edge_lengths(METHODS[m](V, F), F)))


# ----------------------------------------------------------------------------------------
# SLIM
# ----------------------------------------------------------------------------------------
def test_slim_iterations_match_libigl():
    igl = pytest.importorskip("igl")
    V, F = spherical_cap(14, np.radians(80))
    UV0 = slim.initialization(V, F)
    ours = slim.flatten(V, F, UV0=UV0, fixed_iterations=10)
    data = igl.slim_precompute(np.asfortranarray(V), np.asfortranarray(F.astype(np.int32)), np.asfortranarray(UV0),
                               igl.SYMMETRIC_DIRICHLET, np.zeros(0, dtype=np.int32), np.zeros((0, 2)), 0.0)
    ref = igl.slim_solve(data, 10)
    assert np.abs(ours - ref).max() < 1e-9 * np.abs(ref).max()


def test_slim_has_the_lowest_symmetric_dirichlet_energy():
    V, F = spherical_cap(14, np.radians(80))
    Dx, Dy = gradient_operators(V, F)
    area = face_areas_from_lengths(edge_lengths(V, F))

    def energy_at_best_scale(UV):  # the energy is not scale invariant: E(s) = a s^2 + c / s^2
        J = slim.jacobians(Dx, Dy, UV)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        if np.any(det <= 0):
            return np.inf
        frob = np.sum(J ** 2, axis=(1, 2))
        return 2 * np.sqrt(np.sum(area * frob) * np.sum(area * frob / det ** 2))

    best = energy_at_best_scale(slim.flatten(V, F))
    for m in ["LSCM", "BFF", "CETM", "SCP", "ABF++", "ARAP", "MDS", "DEM", "Floater"]:
        assert best < energy_at_best_scale(METHODS[m](V, F))


# ----------------------------------------------------------------------------------------
# MDS
# ----------------------------------------------------------------------------------------
def test_landmark_mds_with_every_vertex_as_landmark_is_classical_mds():
    V, F = spherical_cap(5, np.radians(70))
    n = len(V)
    idx, D = mds.geodesic_landmarks(V, F, n)
    assert sorted(idx) == list(range(n))
    D_full = D[np.argsort(idx)]                      # rows in vertex order
    D_full = 0.5 * (D_full + D_full.T)
    H = np.eye(n) - 1.0 / n
    vals, vecs = np.linalg.eigh(-0.5 * H @ D_full ** 2 @ H)
    ref = vecs[:, -2:] * np.sqrt(vals[-2:])
    assert orthogonal_residual(mds.landmark_mds(D, idx), ref) < 1e-8


def test_geodesic_landmarks_are_distinct_and_carry_their_distances():
    V, F = spherical_cap(8, np.radians(60))
    idx, D = mds.geodesic_landmarks(V, F, 20)
    assert len(np.unique(idx)) == 20 and D.shape == (20, len(V))
    assert np.allclose(D[np.arange(20), idx], 0)


# ----------------------------------------------------------------------------------------
# DEM
# ----------------------------------------------------------------------------------------
def test_dem_equalizes_a_prescribed_density():
    P, F = grid_mesh(30, 30, 1.0, 1.0)
    V = np.c_[P, np.zeros(len(P))]
    inside = np.linalg.norm(P[F].mean(axis=1) - 0.5, axis=1) < 0.25
    population = face_areas_3d(V, F) * np.where(inside, 2.0, 1.0)  # density 2 inside a disk, 1 outside
    UV = dem.density_equalizing_map(P, F, population, P)
    a = face_areas_from_lengths(edge_lengths(UV, F))
    rho = (population / population.sum()) / (a / a.sum())
    q1, median, q3 = np.percentile(rho, [25, 50, 75])
    assert abs(median - 1) < 0.03 and q3 - q1 < 0.08          # cf. Choi & Rycroft, Table 5.1
    assert a[inside].sum() / a.sum() > 1.6 * face_areas_3d(V, F)[inside].sum() / face_areas_3d(V, F).sum()


def test_dem_is_nearly_area_preserving_on_a_curved_surface():
    V, F = spherical_cap(16, np.radians(75))
    log_area = np.abs(log_area_ratio(V, F, dem.flatten(V, F)))
    assert log_area.mean() < 0.03 and np.median(log_area) < 0.03


def test_dem_boundary_is_convex_and_sea_is_an_oriented_disk():
    V, F = spherical_cap(10, np.radians(70))
    b = boundary_loop(F)
    poly = dem.curvature_boundary(V[b])
    e = np.roll(poly, -1, axis=0) - poly
    e_next = np.roll(e, -1, axis=0)
    assert np.all(e[:, 0] * e_next[:, 1] - e[:, 1] * e_next[:, 0] > 0)   # convex, counterclockwise
    assert abs(np.linalg.norm(e, axis=1).sum() / np.linalg.norm(V[np.roll(b, -1)] - V[b], axis=1).sum() - 1) < 1e-3
    verts, faces = dem.construct_sea(dem.initial_map(V, F), F)
    assert np.array_equal(faces[:len(F)], F)
    check_disk(verts, faces)
    assert np.all(signed_areas_2d(verts, faces) > 0)


# ----------------------------------------------------------------------------------------
# Floater
# ----------------------------------------------------------------------------------------
def test_mean_value_weights_reproduce_a_planar_mesh():
    P, F = disk_mesh(8)
    V = np.c_[P, np.zeros(len(P))]
    b = boundary_loop(F)
    UV = harmonic_extension(floater.mean_value_laplacian(V, F), b, P[b])
    assert np.abs(UV - P).max() < 1e-12


def test_floater_fixes_the_boundary_on_a_circle():
    V, F = spherical_cap(10, np.radians(70))
    b = boundary_loop(F)
    UV = floater.flatten(V, F)
    radius = np.linalg.norm(UV[b], axis=1)
    assert np.allclose(radius, radius[0]) and signed_areas_2d(UV, F).min() > 0

