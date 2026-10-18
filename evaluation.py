"""Evaluation of a flattening as in the paper (Sec. 2.2 and Sec. 3), and coverage paths.

  * area distortion log(A3D / A2D) per face,
  * center-to-boundary geodesic deviation (CBGD, Eq. 26),
  * spiral path generation in the UV domain and barycentric back-projection (Eq. 22),
  * end-effector poses along the path (Eqs. 23-24),
  * validity: flipped triangles and boundary self-intersections (global overlaps),
  * coverage paths with a stepover (spiral and raster) and their 3D coverage and spacing (extension).
"""
import numpy as np
from matplotlib.path import Path
from pygeodesic.geodesic import PyGeodesicAlgorithmExact
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from flattening.common import (corner_angles, edge_lengths, gradient_operators,  # noqa: F401 (re-exported)
                               signed_areas_2d)


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
    weight (then to the lowest face index); faces with near-zero UV area are
    skipped. Returns (faces, weights) with face index -1 for points that lie in
    no triangle.

    A point whose weights are all >= -eps lies in the triangle scaled by 1 + 3 eps
    about its centroid, so only faces with a centroid within the largest
    centroid-to-corner distance (plus a margin) are tested, found with a k-d tree.
    The result is the same as testing every face.
    """
    T = UV[F]
    area2 = _cross2(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    usable = np.flatnonzero(np.abs(area2) > 1e-12 * np.abs(area2).max())
    centroid = T[usable].mean(axis=1)
    reach = (1 + 1e-3) * np.linalg.norm(T[usable] - centroid[:, None], axis=2).max()
    faces = np.full(len(points), -1)
    weights = np.zeros((len(points), 3))
    for k, (p, near) in enumerate(zip(points, cKDTree(centroid).query_ball_point(points, reach))):
        if not near:
            continue
        f = usable[np.sort(near)]  # in index order, so ties resolve as in a scan of all faces
        A, B, C = T[f, 0], T[f, 1], T[f, 2]
        wa = _cross2(B - p, C - p) / area2[f]
        wb = _cross2(C - p, A - p) / area2[f]
        W = np.stack([wa, wb, 1.0 - wa - wb], axis=1)
        best = int(np.argmax(W.min(axis=1)))
        if W[best].min() >= -eps:
            w = np.clip(W[best], 0.0, None)
            faces[k], weights[k] = f[best], w / w.sum()
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


def spiral_curve(polygon, center, turns, margin=0.98, samples_per_turn=1000, closing_turns=0):
    """Spiral from `center` out to the boundary as a fine polyline.

    The radius grows linearly with the polar angle up to `margin` times the
    distance to the boundary in that direction, so the outer turn follows the
    boundary shape; it then stays there for `closing_turns` more turns.
    """
    total = turns + closing_turns
    phi = np.linspace(0.0, 2 * np.pi * total, samples_per_turn * total + 1)
    reach = ray_distances(center, polygon, np.mod(phi, 2 * np.pi))
    end = 2 * np.pi * turns
    r = margin * reach * np.minimum(phi, end) / end
    return center + r[:, None] * np.stack([np.cos(phi), np.sin(phi)], axis=1)


def resample(curve, step):
    """Points at constant arc length `step` along a polyline, and its length."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
    q = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(q, s, c) for c in curve.T], axis=1), s[-1]


def spiral_path(polygon, center, turns=3, step=3.2, margin=0.98, samples_per_turn=1000):
    """The paper's spiral path: spiral_curve resampled at constant arc length. Returns (via points, length)."""
    return resample(spiral_curve(polygon, center, turns, margin, samples_per_turn), step)


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


# ----------------------------------------------------------------------------------------
# Coverage paths (extension): planned in 2D with a stepover, measured on the surface
# ----------------------------------------------------------------------------------------
def principal_axis(UV, F):
    """Unit direction of the major principal axis of the 2D map, from the second moments of its
    triangles (so it does not depend on how the vertices are spread); first nonzero component > 0."""
    a = np.abs(signed_areas_2d(UV, F))
    T = UV[F]
    g = T.mean(axis=1)                          # triangle centroids
    d, c = T - g[:, None], g - a @ g / a.sum()  # corners about the centroids, centroids about the center
    M = np.einsum("f,fki,fkj->ij", a / 12, d, d) + np.einsum("f,fi,fj->ij", a, c, c)
    e = np.linalg.eigh(M)[1][:, -1]
    return e if (e[0], e[1]) > (0, 0) else -e


def raster_passes(polygon, axis, spacing):
    """Raster (boustrophedon) path: lines along `axis`, `spacing` apart, clipped to the polygon.

    The lines are centered on the polygon's extent across `axis`, so the outer ones are at most
    spacing / 2 from its extremes, and every other line runs backwards. Returns the passes in the
    order the tool runs them, each a (2, 2) array of its start and end point, and the line index
    of each pass; a line that crosses a concave part of the polygon gives several passes.
    """
    normal = np.array([-axis[1], axis[0]])
    x, y = polygon @ axis, polygon @ normal
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    count = int(np.ceil((y.max() - y.min()) / spacing))
    passes, lines = [], []
    for j in range(count):
        c = 0.5 * (y.max() + y.min()) + (j - 0.5 * (count - 1)) * spacing
        cross = (y <= c) != (y2 <= c)  # boundary edges that straddle the line
        ends = np.sort(x[cross] + (c - y[cross]) / (y2[cross] - y[cross]) * (x2[cross] - x[cross]))
        ends = ends.reshape(-1, 2)     # (entry, exit) pairs along the line
        if j % 2:
            ends = ends[::-1, ::-1]
        passes += [np.outer(e, axis) + c * normal for e in ends]
        lines += [j] * len(ends)
    return passes, np.array(lines)


def coverage_spiral(polygon, center, spacing, margin=0.98, samples_per_turn=1000):
    """Spiral (spiral_curve) with enough turns to keep them at most `spacing` apart in every
    direction, closed by one more turn along the boundary; returned as a list of one pass."""
    reach = ray_distances(center, polygon, np.linspace(0.0, 2 * np.pi, samples_per_turn, endpoint=False))
    turns = int(np.ceil(margin * reach.max() / spacing))
    return [spiral_curve(polygon, center, turns, margin, samples_per_turn, closing_turns=1)]


def next_line_distance(points, line):
    """Distance from each point to the nearest point on the next raster line (line indices are
    non-decreasing along the path); NaN on the last line."""
    lo = np.searchsorted(line, line + 1)
    hi = np.searchsorted(line, line + 1, side="right")
    d = np.full(len(points), np.nan)
    for i in np.flatnonzero(hi > lo):
        d[i] = np.linalg.norm(points[lo[i]:hi[i]] - points[i], axis=1).min()
    return d


def subdivision_barycentrics(n):
    """Barycentric coordinates of the centroids of the n * n equal triangles of a regular subdivision."""
    i, j = (x.ravel() for x in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    w = np.vstack([np.c_[i + 1 / 3, j + 1 / 3][i + j <= n - 1], np.c_[i + 2 / 3, j + 2 / 3][i + j <= n - 2]]) / n
    return np.c_[w, 1 - w.sum(axis=1)]


def length_within_tolerance(V, F, UV, tol=0.2):
    """Faces on which the map scales lengths in every direction to within +-tol of its
    area-equivalent scale sqrt(A2D / A3D); flipped faces never count."""
    Dx, Dy = gradient_operators(V, F)
    J = np.stack([np.stack([Dx @ UV[:, c], Dy @ UV[:, c]], axis=1) for c in range(2)], axis=1)
    sigma = np.linalg.svd(J, compute_uv=False)  # largest and smallest 2D length per unit 3D length
    k = np.sqrt(np.abs(signed_areas_2d(UV, F)).sum() / face_areas_3d(V, F).sum())
    return (np.linalg.det(J) > 0) & (sigma[:, 0] <= k / (1 - tol)) & (sigma[:, 1] >= k / (1 + tol))


def _covered_share(X, F, bary, path, radius):
    """Share of each face of (X, F) within `radius` of the path points (sampled at `bary`)."""
    samples = np.einsum("sk,fkd->fsd", bary, X[F]).reshape(-1, X.shape[1])
    near = np.isfinite(cKDTree(path).query(samples, distance_upper_bound=radius)[0])
    return near.reshape(len(F), -1).mean(axis=1)


def coverage_measures(V, F, UV, passes, spacing, tool_width=1.2, tol=0.2, lines=None):
    """Back-project a coverage path planned in the 2D domain and measure it on the surface.

    `passes` are polylines in the order the tool runs them and `spacing` is the stepover
    s of the plan. The surface stepover planned from it is k s, with the map's
    area-equivalent scale k = sqrt(A3D / A2D), and the tool covers a band of width
    tool_width * k * s (straight-line distance, i.e. a ball-shaped footprint). The path
    is sampled every s / 10 and has a via point every s / 2. Returns (measures, arrays);
    the measures are
      coverage_pct           share of the surface area within the tool radius of the path
      planned_coverage_pct   share of the 2D area within the tool radius in the plan
      step_*                 3D distance between consecutive via points of a pass relative to k
                             times their 2D distance: mean |error| and share off by more than tol
      relative_length        3D path length over that of an ideal raster, A3D / (k s)
      stepover_*             only with the raster line index of each pass (`lines`): 3D distance to
                             the next line relative to k times the 2D one, where that is at most
                             1.5 s (a shorter next pass can end farther away): mean |error| and the
                             shares of the path more than tol too wide (gaps) or too narrow (overlap)
    """
    a3, a2 = face_areas_3d(V, F), np.abs(signed_areas_2d(UV, F))
    k = np.sqrt(a3.sum() / a2.sum())
    radius = 0.5 * tool_width * spacing  # in the plan; k * radius on the surface
    sampled = [resample(p, spacing / 10)[0] for p in passes]
    pass_id = np.repeat(np.arange(len(sampled)), [len(p) for p in sampled])
    is_via = np.concatenate([np.arange(len(p)) % 5 == 0 for p in sampled])
    P2 = np.vstack(sampled)
    faces, weights = locate(UV, F, P2)
    found = faces >= 0
    P3 = np.full((len(P2), 3), np.nan)
    P3[found] = back_project(V, F, faces[found], weights[found])

    via = np.flatnonzero(is_via)  # along the path
    a, b = via[:-1], via[1:]
    keep = (pass_id[a] == pass_id[b]) & found[a] & found[b]
    a, b = a[keep], b[keep]
    step = np.linalg.norm(P3[b] - P3[a], axis=1) / (k * np.linalg.norm(P2[b] - P2[a], axis=1)) - 1

    bary = subdivision_barycentrics(int(np.ceil(8 * edge_lengths(V, F).mean() / (k * radius))))
    covered = _covered_share(V, F, bary, P3[found], k * radius)
    planned = _covered_share(UV, F, bary, P2, radius)
    segment = (pass_id[1:] == pass_id[:-1]) & found[1:] & found[:-1]
    length = np.linalg.norm(np.diff(P3, axis=0)[segment], axis=1).sum()
    measures = {
        "passes": len(passes),
        "unmapped": int(np.sum(~found)),
        "coverage_pct": float(100 * a3 @ covered / a3.sum()),
        "planned_coverage_pct": float(100 * a2 @ planned / a2.sum()),
        "step_mean_abs_dev_pct": float(100 * np.abs(step).mean()),
        "step_off_pct": float(100 * np.mean(np.abs(step) > tol)),
        "relative_length": float(length * k * spacing / a3.sum()),
    }
    arrays = dict(points_2d=P2, points_3d=P3, pass_id=pass_id, uncovered=1 - covered)
    if lines is not None:  # across the path
        line = lines[pass_id][found]
        d2, d3 = next_line_distance(P2[found], line), next_line_distance(P3[found], line)
        pair = d2 <= 1.5 * spacing  # False where NaN
        stepover = np.full(len(P2), np.nan)
        stepover[np.flatnonzero(found)[pair]] = d3[pair] / (k * d2[pair]) - 1
        over = stepover[np.isfinite(stepover)]
        measures.update(stepover_mean_abs_dev_pct=float(100 * np.abs(over).mean()),
                        stepover_wide_pct=float(100 * np.mean(over > tol)),
                        stepover_narrow_pct=float(100 * np.mean(over < -tol)))
        arrays["stepover"] = stepover
    return measures, arrays
