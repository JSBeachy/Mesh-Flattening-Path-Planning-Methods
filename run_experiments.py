"""Run the flattening methods on every test mesh and evaluate them (paper Sec. 3).

Writes results/summary.csv (one row per object and method) and the per-run
arrays in results/data/*.npz that make_figures.py plots.

Usage:
  python run_experiments.py                       all methods
  python run_experiments.py --methods LSCM,SCP    only these; the other rows of summary.csv are kept
"""
import argparse
import csv
import os
import time

import numpy as np

import evaluation as ev
from flattening import FAMILY, METHODS
from flattening.common import boundary_loop, load_mesh

ROOT = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(ROOT, "meshes")
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(RESULTS, "data")

# (file name, display name, object of the paper it stands in for)
OBJECTS = [
    ("wash_basin", "Wash basin", "Wash basin"),
    ("turtle", "Turtle", "Turtle"),
    ("face_mask", "Face mask", "Face mask"),
    ("bump_cap", "Bump cap", "Bumpcap"),
    ("animal_head", "Animal head", "Kitten face"),
    ("lilium_top", "Lilium top", "Lilium top"),
    ("four_sided_star", "Four-sided star", "Four-sided star"),
    ("car_body", "Car body", "Car body"),
]
REPEAT_BELOW = 60.0      # methods faster than this (seconds) are timed 3 times and the median is reported
UV_FRAME = 100.0         # path planning frame: UV scaled to [0, 100] (paper Sec. 2.2)
SPIRAL_TURNS, SPIRAL_STEP = 3, 3.2


def evaluate(V, F, boundary, geo, p_true, UV):
    """All measures for one flattening; returns (summary row, arrays to save)."""
    uv = ev.normalize_uv(UV)  # unit box, as in the paper's plots
    signed = ev.signed_areas_2d(uv, F)
    log_area = ev.log_area_ratio(V, F, uv)

    center_2d, center_3d, p_mapped = ev.mapped_center(V, F, uv, boundary)
    delta, d_true, d_mapped = ev.cbgd(geo, boundary, p_true, p_mapped)

    # Spiral path in the [0, 100] frame, back-projected with barycentric weights (Eq. 22).
    uv100 = uv * UV_FRAME
    via, length = ev.spiral_path(uv100[boundary], center_2d * UV_FRAME, SPIRAL_TURNS, SPIRAL_STEP)
    faces, weights = ev.locate(uv100, F, via)
    found = faces >= 0
    path_3d = ev.back_project(V, F, faces[found], weights[found])
    pose_positions, pose_quaternions = ev.tool_poses(V, F, faces[found], path_3d)
    # 3D spacing of consecutive via points relative to the nominal step (2D step times the
    # area-equivalent scale of the map); only pairs of successfully mapped neighbors count.
    nominal = SPIRAL_STEP * np.sqrt(ev.face_areas_3d(V, F).sum() / np.abs(ev.signed_areas_2d(uv100, F)).sum())
    pairs = found[:-1] & found[1:]
    idx = np.cumsum(found) - 1
    steps = np.linalg.norm(path_3d[idx[1:][pairs]] - path_3d[idx[:-1][pairs]], axis=1)
    spacing_dev = steps / nominal - 1

    row = {
        "flipped_faces": int(np.sum(signed <= 0)),
        "boundary_area_positive": bool(signed.sum() > 0),  # paper's validity check (i)
        "boundary_self_intersections": ev.boundary_self_intersections(uv, boundary),
        "mean_abs_log_area": float(np.abs(log_area).mean()),
        "mean_abs_angle_deg": float(ev.angle_distortion_deg(V, F, uv)),
        "cbgd_mean_pct": float(100 * np.abs(delta).mean()),
        "cbgd_max_pct": float(100 * np.abs(delta).max()),
        "mapped_center": p_mapped,
        "center_offset_pct": float(100 * geo([p_true])[p_mapped] / d_true.mean()),
        "spiral_length": float(length),
        "spiral_via_points": int(len(via)),
        "spiral_unmapped": int(np.sum(~found)),
        "spacing_mean_abs_dev_pct": float(100 * np.abs(spacing_dev).mean()),
        "spacing_over_20pct": float(100 * np.mean(np.abs(spacing_dev) > 0.2)),
    }
    arrays = dict(uv=uv, log_area=log_area, center_2d=center_2d, center_3d=center_3d, p_mapped=p_mapped,
                  delta=delta, d_true=d_true, d_mapped=d_mapped, via=via, via_found=found,
                  path_3d=path_3d, spacing_dev=spacing_dev, pose_positions=pose_positions,
                  pose_quaternions=pose_quaternions)
    return row, arrays


def run_method(flatten, V, F):
    """Flatten and time it: three runs (median reported) unless the first takes REPEAT_BELOW seconds or more."""
    times = []
    while True:
        t0 = time.perf_counter()
        UV = flatten(V, F)
        times.append(time.perf_counter() - t0)
        if len(times) == 3 or times[0] >= REPEAT_BELOW:
            return UV, times


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", default=",".join(METHODS), help="comma-separated methods (default: all)")
    chosen = {m for m in parser.parse_args().methods.split(",") if m}
    if chosen - set(METHODS):
        raise SystemExit(f"unknown method(s): {', '.join(sorted(chosen - set(METHODS)))}; "
                         f"choose from {', '.join(METHODS)}")
    methods = [m for m in METHODS if m in chosen]

    os.makedirs(DATA, exist_ok=True)
    summary = os.path.join(RESULTS, "summary.csv")
    rows = {}
    if os.path.exists(summary):  # rows of methods that are not run now are kept
        with open(summary) as f:
            rows = {(r["object"], r["method"]): r for r in csv.DictReader(f)}

    for key, name, paper_name in OBJECTS:
        path = os.path.join(MESH_DIR, key + ".obj")
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found - run `python make_meshes.py` first")
        V, F = load_mesh(path)
        boundary = boundary_loop(F)
        geo = ev.Geodesics(V, F)
        p_true = ev.true_center(geo, boundary)
        np.savez_compressed(os.path.join(DATA, f"{key}__mesh.npz"), V=V, F=F, boundary=boundary, p_true=p_true,
                            s=ev.boundary_parameter(V, boundary))
        print(f"{name}: {len(V)} vertices, {len(F)} faces, {len(boundary)} boundary vertices")
        for method in methods:
            UV, times = run_method(METHODS[method], V, F)
            if not np.all(np.isfinite(UV)):
                raise RuntimeError(f"{method} produced non-finite UVs on {name}")
            row, arrays = evaluate(V, F, boundary, geo, p_true, UV)
            np.savez_compressed(os.path.join(DATA, f"{key}__{method}.npz"), times=times, **arrays)
            rows[(name, method)] = {"object": name, "paper_object": paper_name, "method": method,
                                    "family": FAMILY[method], "vertices": len(V), "faces": len(F),
                                    "boundary_vertices": len(boundary), "true_center": p_true,
                                    "time_s": float(np.median(times)), "timing_runs": len(times), **row}
            print(f"  {method:7s} time {np.median(times):8.3f} s | |log area| {row['mean_abs_log_area']:.3f} | "
                  f"angle {row['mean_abs_angle_deg']:5.2f} deg | CBGD {row['cbgd_mean_pct']:6.2f} % "
                  f"(max {row['cbgd_max_pct']:6.2f} %) | flips {row['flipped_faces']} | "
                  f"overlaps {row['boundary_self_intersections']}")

    ordered = [rows[(name, m)] for _, name, _ in OBJECTS for m in METHODS if (name, m) in rows]
    fields = list(next(r for r in ordered if r["method"] in methods))
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)

    print("\nMean +- std over objects:")
    for method in METHODS:
        r = [x for x in ordered if x["method"] == method]
        if len(r) != len(OBJECTS):
            continue
        stats = {k: (np.mean([float(x[k]) for x in r]), np.std([float(x[k]) for x in r]))
                 for k in ["cbgd_mean_pct", "mean_abs_log_area", "mean_abs_angle_deg", "time_s"]}
        print(f"  {method:7s} CBGD {stats['cbgd_mean_pct'][0]:5.2f} +- {stats['cbgd_mean_pct'][1]:5.2f} % | "
              f"|log area| {stats['mean_abs_log_area'][0]:.3f} | angle {stats['mean_abs_angle_deg'][0]:5.2f} deg | "
              f"time {stats['time_s'][0]:.3f} +- {stats['time_s'][1]:.3f} s")


if __name__ == "__main__":
    main()
