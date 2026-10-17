"""Surface flattening by multidimensional scaling (Zigelman, Kimmel & Kiryati 2002).

The flat map is the 2D configuration whose Euclidean distances best match the
geodesic distances on the surface (classical scaling: double-centered squared
distances, top two eigenvectors). Like Zigelman et al., who flatten a subset of
anchor vertices (fast-marching geodesics) and interpolate the rest, we flatten
K landmark vertices, here with exact polyhedral geodesics (MMP) and landmarks
chosen by geodesic farthest-point sampling, and place all vertices by landmark
MDS (de Silva & Tenenbaum 2004), which reproduces classical scaling exactly
when the distances are Euclidean. MDS fixes the map only up to a rigid motion
and a reflection; the reflection is chosen so that the map preserves
orientation. Distance-based, so it does not prevent flipped triangles.
"""
import numpy as np
from pygeodesic.geodesic import PyGeodesicAlgorithmExact

from .common import boundary_loop, orient_ccw


def geodesic_landmarks(V, F, k):
    """k landmark vertices by geodesic farthest-point sampling, and their (k, n) distance rows."""
    alg = PyGeodesicAlgorithmExact(np.ascontiguousarray(V, dtype=float), np.ascontiguousarray(F, dtype=np.int32))

    def dist(i):
        d, _ = alg.geodesicDistances(np.array([i], dtype=np.int32), None)
        return d

    first = int(boundary_loop(F)[0])
    start = int(np.argmax(dist(first)))      # a deterministic, extreme first landmark
    landmarks, rows = [start], [dist(start)]
    nearest = rows[0].copy()
    while len(landmarks) < k:
        i = int(np.argmax(nearest))
        landmarks.append(i)
        rows.append(dist(i))
        nearest = np.minimum(nearest, rows[-1])
    return np.array(landmarks), np.array(rows)


def landmark_mds(D_landmarks, landmarks, dim=2):
    """de Silva & Tenenbaum (2004): embed all points from their distances to k landmarks."""
    D2 = D_landmarks ** 2                           # (k, n) squared distances
    Dk = D2[:, landmarks]                           # (k, k) between landmarks
    k = len(landmarks)
    H = np.eye(k) - 1.0 / k
    vals, vecs = np.linalg.eigh(-0.5 * H @ Dk @ H)
    top = np.argsort(vals)[::-1][:dim]
    if np.any(vals[top] <= 0):
        raise RuntimeError("landmark MDS: fewer than two positive eigenvalues")
    pinv = vecs[:, top] / np.sqrt(vals[top])        # (k, dim)
    return -0.5 * (D2 - Dk.mean(axis=1, keepdims=True)).T @ pinv


def flatten(V, F, landmarks=128):
    k = min(landmarks, len(V))
    idx, D = geodesic_landmarks(V, F, k)
    return orient_ccw(landmark_mds(D, idx), F)
