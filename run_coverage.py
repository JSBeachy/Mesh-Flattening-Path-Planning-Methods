"""Coverage paths planned on every flattening and measured on the surface (extension).

For each object and method saved by run_experiments.py, two coverage paths with the same
stepover are planned in the 2D domain - a spiral from the 2D center and a raster
(boustrophedon) whose passes run along the principal axis of the map - then mapped back
to the surface and measured (evaluation.coverage_measures), together with the share of
the surface on which the map keeps every length within the spacing tolerance
(evaluation.length_within_tolerance).

Writes results/coverage.csv (one row per object and method) and the arrays that
make_figures.py plots in results/data/<object>__<method>__coverage.npz.

Usage:  python run_coverage.py      (after run_experiments.py; ~3 min)
"""
import csv
import os

import numpy as np

import evaluation as ev
from flattening import METHODS
from run_experiments import DATA, OBJECTS, RESULTS

STEPOVER = 0.05    # distance between passes, as a fraction of the square root of the surface area
TOOL_WIDTH = 1.2   # tool footprint in stepovers: 20 % planned overlap between neighboring passes
TOLERANCE = 0.2    # spacing errors the overlap absorbs (the paper's 15-20 % tolerance, upper end)


def main():
    rows = []
    for key, name, _ in OBJECTS:
        mesh = np.load(os.path.join(DATA, f"{key}__mesh.npz"))
        V, F, boundary = mesh["V"], mesh["F"], mesh["boundary"]
        a3 = ev.face_areas_3d(V, F)
        print(name)
        for method in METHODS:
            run = np.load(os.path.join(DATA, f"{key}__{method}.npz"))
            uv = run["uv"]
            # The same surface stepover STEPOVER * sqrt(A3D) for every method, i.e. STEPOVER * sqrt(A2D) in 2D.
            s = STEPOVER * np.sqrt(np.abs(ev.signed_areas_2d(uv, F)).sum())
            within = ev.length_within_tolerance(V, F, uv, TOLERANCE)
            row = {"object": name, "method": method, "within_tolerance_pct": float(100 * a3[within].sum() / a3.sum())}
            arrays = {"within_tolerance": within}
            # The spiral's turns are at most s apart (closer where the boundary is nearer the center than
            # its farthest point), the raster's lines exactly s apart, so only the raster has a stepover error.
            raster, lines = ev.raster_passes(uv[boundary], ev.principal_axis(uv, F), s)
            for path, passes, path_lines in [("spiral", ev.coverage_spiral(uv[boundary], run["center_2d"], s), None),
                                             ("raster", raster, lines)]:
                measures, data = ev.coverage_measures(V, F, uv, passes, s, TOOL_WIDTH, TOLERANCE, path_lines)
                row.update({f"{path}_{k}": v for k, v in measures.items()})
                arrays.update({f"{path}_{k}": v for k, v in data.items()})
            np.savez_compressed(os.path.join(DATA, f"{key}__{method}__coverage.npz"), **arrays)
            rows.append(row)
            print(f"  {method:7s} within tolerance {row['within_tolerance_pct']:5.1f} % | coverage: spiral "
                  f"{row['spiral_coverage_pct']:5.1f} %, raster {row['raster_coverage_pct']:5.1f} % | raster stepover "
                  f"error {row['raster_stepover_mean_abs_dev_pct']:4.1f} %")

    with open(os.path.join(RESULTS, "coverage.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nMean over objects:")
    for method in METHODS:
        r = [x for x in rows if x["method"] == method]
        mean = {k: np.mean([x[k] for x in r]) for k in r[0] if k not in ("object", "method")}
        print(f"  {method:7s} within tolerance {mean['within_tolerance_pct']:5.1f} % | coverage: spiral "
              f"{mean['spiral_coverage_pct']:5.1f} %, raster {mean['raster_coverage_pct']:5.1f} % | raster stepover "
              f"error {mean['raster_stepover_mean_abs_dev_pct']:4.1f} %")


if __name__ == "__main__":
    main()
