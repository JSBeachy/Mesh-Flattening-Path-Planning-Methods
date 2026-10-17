"""Evaluation of a flattening as in the paper (Sec. 2.2 and Sec. 3).

  * area distortion log(A3D / A2D) per face,
  * center-to-boundary geodesic deviation (CBGD, Eq. 26),
  * spiral path generation in the UV domain and barycentric back-projection (Eq. 22),
  * end-effector poses along the path (Eqs. 23-24),
  * validity: flipped triangles and boundary self-intersections (global overlaps).
"""
import numpy as np
from matplotlib.path import Path
from pygeodesic.geodesic import PyGeodesicAlgorithmExact
from scipy.spatial.transform import Rotation

from flattening.common import corner_angles, edge_lengths, signed_areas_2d  # noqa: F401 (re-exported)


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def face_areas_3d(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def log_area_ratio(V, F, UV):
    """Per-face area distortion log(A3D / A2D), both domains scaled to unit total area."""
    a3 = face_areas_3d(V, F)
    a2 = np.abs(signed_areas_2d(UV, F))
    return np.log((a3 / a3.sum()) / (a2 / a2.sum()))


def angle_distortion_deg(V, F, UV):
    """Mean absolute difference between 3D and 2D corner angles, in degrees (extra metric)."""
    return np.degrees(np.abs(corner_angles(edge_lengths(UV, F)) - corner_angles(edge_lengths(V, F)))).mean()


def boundary_self_intersections(UV, boundary):
    """Number of pairs of non-adjacent boundary edges that cross in the UV domain (0: simple polygon).

    Together with the flipped-triangle count this decides global injectivity: a
    map without flipped triangles whose boundary is a simple polygon is one-to-one.
    """
    a, b = UV[boundary], UV[np.roll(boundary, -1)]
    nb = len(boundary)
    j = np.arange(nb)
    count = 0
    for s in range(0, nb, 256):
        i = np.arange(s, min(s + 256, nb))
        p, q = a[i, None], b[i, None]
        d1, d2 = _cross2(q - p, a[None] - p), _cross2(q - p, b[None] - p)
        d3, d4 = _cross2(b[None] - a[None], p - a[None]), _cross2(b[None] - a[None], q - a[None])
        gap = np.abs(i[:, None] - j[None])
        candidate = (j[None] > i[:, None]) & (gap > 1) & (gap < nb - 1)
        count += int(np.sum((d1 * d2 < 0) & (d3 * d4 < 0) & candidate))
    return count


def normalize_uv(UV, size=1.0):
    """Translate and uniformly scale UV so the bounding box starts at 0 and its larger side is `size`."""
    lo = UV.min(axis=0)
    return (UV - lo) * (size / (UV.max(axis=0) - lo).max())


def boundary_parameter(V, boundary):
    """Normalized arc length in [0, 1) of each boundary vertex along the 3D boundary loop."""
    seg = np.linalg.norm(V[np.roll(boundary, -1)] - V[boundary], axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)[:-1]]) / seg.sum()


# ----------------------------------------------------------------------------------------
# Geodesic distances and the two centers
# ----------------------------------------------------------------------------------------
class Geodesics:
    """Exact polyhedral geodesic distances on M (MMP algorithm, via pygeodesic)."""

    def __init__(self, V, F):
        self._alg = PyGeodesicAlgorithmExact(np.ascontiguousarray(V, dtype=float),
                                             np.ascontiguousarray(F, dtype=np.int32))

    def __call__(self, sources):
        """Distance from every vertex to the nearest of the given source vertices."""
        d, _ = self._alg.geodesicDistances(np.asarray(sources, dtype=np.int32), None)
        return d


def true_center(geo, boundary):
    """P_t: the vertex with the maximum geodesic distance from the boundary."""
    return int(np.argmax(geo(boundary)))


def _distance_to_segments(p, a, b):
    """Distance from each point p to the closest of the segments (a_k, b_k)."""
    ab = b - a
    ab2 = np.einsum("ij,ij->i", ab, ab)
    out = np.empty(len(p))
    chunk = max(1, 2_000_000 // len(a))
    for s in range(0, len(p), chunk):
        q = p[s:s + chunk, None, :] - a[None]
        t = np.clip(np.einsum("cki,ki->ck", q, ab) / ab2, 0.0, 1.0)
        out[s:s + chunk] = np.linalg.norm(q - t[..., None] * ab[None], axis=2).min(axis=1)
    return out


def pole_of_inaccessibility(polygon, rel_tol=1e-4):
    """Point inside a polygon farthest from its boundary (polylabel branch and bound).

    The returned point's boundary distance is within rel_tol * (bounding box size)
    of the maximum.
    """
    a, b = polygon, np.roll(polygon, -1, axis=0)
    inside = Path(polygon)

    def signed_distance(p):
        d = _distance_to_segments(p, a, b)
        return np.where(inside.contains_points(p), d, -d)

    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    tol = rel_tol * (hi - lo).max()
    # Seed with the area centroid, then refine square cells of half size h.
    cross = _cross2(a, b)
    best = ((a + b) * cross[:, None]).sum(axis=0) / (3 * cross.sum())
    best_d = signed_distance(best[None])[0]
    h = 0.5 * (hi - lo).min()
    xs, ys = np.arange(lo[0] + h, hi[0] + h, 2 * h), np.arange(lo[1] + h, hi[1] + h, 2 * h)
    cells = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    children = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    while len(cells):
        d = signed_distance(cells)
        k = np.argmax(d)
        if d[k] > best_d:
            best, best_d = cells[k], d[k]
        cells = cells[d + h * np.sqrt(2) > best_d + tol]  # cells that may still contain a better point
        h /= 2
        cells = (cells[:, None, :] + h * children[None]).reshape(-1, 2)
    return best


def locate(UV, F, points, eps=1e-9):
    """Find the UV triangle containing each point and its barycentric weights (paper Sec. 2.2).

    Weights >= -eps are accepted, small negatives are clamped and the weights
    renormalized; ties on shared edges go to the face with the largest minimum
    weight; faces with near-zero UV area are skipped. Returns (faces, weights)
    with face index -1 for points that lie in no triangle.
    """
    A, B, C = UV[F[:, 0]], UV[F[:, 1]], UV[F[:, 2]]
    area2 = _cross2(B - A, C - A)
    usable = np.abs(area2) > 1e-12 * np.abs(area2).max()
    area2 = np.where(usable, area2, 1.0)
    faces = np.full(len(points), -1)
    weights = np.zeros((len(points), 3))
    for k, p in enumerate(points):
        wa = _cross2(B - p, C - p) / area2
        wb = _cross2(C - p, A - p) / area2
        W = np.stack([wa, wb, 1.0 - wa - wb], axis=1)
        worst = np.where(usable, W.min(axis=1), -np.inf)
        f = int(np.argmax(worst))
        if worst[f] >= -eps:
            w = np.clip(W[f], 0.0, None)
            faces[k], weights[k] = f, w / w.sum()
    return faces, weights


def back_project(V, F, faces, weights):
    """Eq. (22): P_M = sum_k W_k V_k over the corresponding 3D triangle."""
    return np.einsum("qk,qkd->qd", weights, V[F[faces]])


def mapped_center(V, F, UV, boundary):
    """Back-project the 2D pole of inaccessibility to M.

    Returns the 2D center, its 3D back-projection and P_m, the vertex of the
    containing triangle closest to that 3D point (geodesics are vertex-sourced,
    like P_t).
    """
    center_2d = pole_of_inaccessibility(UV[boundary])
    faces, weights = locate(UV, F, center_2d[None])
    if faces[0] < 0:
        raise RuntimeError("2D center does not lie inside the parameterization")
    point_3d = back_project(V, F, faces, weights)[0]
    corners = F[faces[0]]
    vertex = int(corners[np.argmin(np.linalg.norm(V[corners] - point_3d, axis=1))])
    return center_2d, point_3d, vertex


def cbgd(geo, boundary, p_true, p_mapped):
    """Eq. (26): relative deviation of the boundary distances from the mapped vs. the true center."""
    d_true = geo([p_true])[boundary]
    d_mapped = geo([p_mapped])[boundary]
    return (d_mapped - d_true) / d_true, d_true, d_mapped


# ----------------------------------------------------------------------------------------
# Spiral path (paper Sec. 2.2)
# ----------------------------------------------------------------------------------------
def ray_distances(center, polygon, angles):
    """Distance from `center` to the first polygon crossing along each direction."""
    a = polygon - center
    e = np.roll(polygon, -1, axis=0) - polygon
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    denom = _cross2(d[:, None, :], e[None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross2(a[None, :, :], e[None, :, :]) / denom
        s = _cross2(a[None, :, :], d[:, None, :]) / denom
    t = np.where((t > 0) & (s >= 0) & (s <= 1), t, np.inf)
    return t.min(axis=1)


def spiral_path(polygon, center, turns=3, step=3.2, margin=0.98, samples_per_turn=1000):
    """Contour-parallel spiral from `center` out to the boundary, resampled at constant arc length.

    The radius grows linearly with the polar angle up to `margin` times the
    distance to the boundary in that direction, so the outer turn follows the
    boundary shape.
    """
    phi = np.linspace(0.0, 2 * np.pi * turns, samples_per_turn * turns + 1)
    reach = ray_distances(center, polygon, np.mod(phi, 2 * np.pi))
    r = margin * reach * phi / phi[-1]
    curve = center + r[:, None] * np.stack([np.cos(phi), np.sin(phi)], axis=1)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
    q = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(q, s, curve[:, 0]), np.interp(q, s, curve[:, 1])], axis=1), s[-1]


def tool_poses(V, F, faces, points, offset=0.0):
    """End-effector poses along a back-projected path (paper Eqs. 23-24).

    N_i is the unit normal of the facet containing P_i (the only facet incident
    to an interior point), T_i the direction to the next via point projected
    onto the tangent plane. The tool z axis is -N_i. The paper writes
    R_i = [T_i, N_i x T_i, -N_i], whose determinant is -1 (a reflection); we use
    the proper rotation R_i = [T_i, T_i x N_i, -N_i]. Returns the offset
    positions P_i + offset * N_i and the quaternions (x, y, z, w).
    """
    n = np.cross(V[F[faces, 1]] - V[F[faces, 0]], V[F[faces, 2]] - V[F[faces, 0]])
    N = n / np.linalg.norm(n, axis=1, keepdims=True)
    d = np.diff(points, axis=0, append=2 * points[-1:] - points[-2:-1])  # last point reuses the last step
    T = d - np.sum(d * N, axis=1, keepdims=True) * N
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    R = np.stack([T, np.cross(T, N), -N], axis=2)
    return points + offset * N, Rotation.from_matrix(R).as_quat()
