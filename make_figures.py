"""Figures and tables from the saved results (counterparts of the paper's Figs. 1-12 and Table 2).

Usage:  python make_figures.py      (after run_experiments.py)
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from flattening import METHODS  # noqa: E402
from run_experiments import DATA, OBJECTS, RESULTS, SPIRAL_STEP  # noqa: E402

FIGURES = os.path.join(RESULTS, "figures")

# Categorical slots 1-4 of the validated reference palette, in fixed order (one color per method).
COLORS = {"LSCM": "#2a78d6", "BFF": "#eb6834", "CETM": "#1baf7a", "ARAP": "#eda100"}
MARKERS = {"LSCM": "o", "BFF": "s", "CETM": "D", "ARAP": "^"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# Diverging blue <-> red with a neutral gray midpoint (for signed log ratios and deviations).
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f4b0ae", "#e34948", "#931f1f"])

# Display orientation: (elevation, azimuth) in degrees; the two scans are y-up and are turned z-up.
VIEWS = {"wash_basin": (40, -60), "turtle": (50, -100), "face_mask": (10, -75), "bump_cap": (25, -40),
         "animal_head": (25, -60), "lilium_top": (45, -60), "four_sided_star": (35, -60),
         "car_body": (35, -60)}
Y_UP = {"face_mask", "animal_head"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": SURFACE, "savefig.facecolor": "white",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 8, "text.color": INK, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "axes.spines.top": False, "axes.spines.right": False,
})


def load(key):
    mesh = dict(np.load(os.path.join(DATA, f"{key}__mesh.npz")))
    runs = {m: dict(np.load(os.path.join(DATA, f"{key}__{m}.npz"))) for m in METHODS}
    return mesh, runs


def read_summary():
    with open(os.path.join(RESULTS, "summary.csv")) as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------------------
def view_frame(key):
    elev, azim = np.radians(VIEWS[key])
    toward_camera = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    right = np.array([-np.sin(azim), np.cos(azim), 0.0])
    return right, np.cross(toward_camera, right), toward_camera


def display_coords(key, V):
    return V[:, [0, 2, 1]] * np.array([1, -1, 1]) if key in Y_UP else V


def render_mesh(ax, key, V, F, edges=False):
    """Orthographic, Lambert-shaded rendering with the painter's algorithm; returns the projector."""
    right, up, toward = view_frame(key)
    W = display_coords(key, V)
    n = np.cross(W[F[:, 1]] - W[F[:, 0]], W[F[:, 2]] - W[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    light = toward + 0.6 * up
    shade = 0.35 + 0.65 * np.abs(n @ (light / np.linalg.norm(light)))
    rgb = np.repeat(shade[:, None] * 0.93, 3, axis=1)
    order = np.argsort((W @ toward)[F].mean(axis=1))

    def project(X):
        Y = display_coords(key, X)
        return np.stack([Y @ right, Y @ up], axis=1)

    P = project(V)
    ax.add_collection(PolyCollection(P[F[order]], facecolors=rgb[order],
                                     edgecolors=(0, 0, 0, 0.12) if edges else "face",
                                     linewidths=0.15 if edges else 0.3))
    ax.set_xlim(P[:, 0].min(), P[:, 0].max())
    ax.set_ylim(P[:, 1].min(), P[:, 1].max())
    ax.set_aspect("equal")
    ax.axis("off")
    return project


def plot_uv(ax, uv, F, values, edges=False):
    pc = PolyCollection(uv[F], array=values, cmap=DIVERGING, norm=Normalize(-1, 1),
                        edgecolors=(0, 0, 0, 0.15) if edges else "face", linewidths=0.1 if edges else 0.2)
    ax.add_collection(pc)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.grid(False)
    return pc


def method_legend(ax, **kw):
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=m) for m in METHODS]
    ax.legend(handles=handles, **kw)


# ----------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------
def figure_object(key, name):
    """Counterpart of Figs. 2-9: area distortion maps and center-to-boundary geodesic distances."""
    mesh, runs = load(key)
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.15, 1, 0.8], width_ratios=[1, 1, 1, 1, 0.05],
                          hspace=0.35, wspace=0.3)
    ax = fig.add_subplot(gs[0, 1:3])
    render_mesh(ax, key, mesh["V"], mesh["F"], edges=len(mesh["F"]) < 20000)
    ax.set_title(f"{name}  ({len(mesh['V'])} vertices, {len(mesh['F'])} faces)", fontsize=13)
    s, d0 = mesh["s"], runs["LSCM"]["d_true"].mean()
    distance_axes = []
    for k, m in enumerate(METHODS):
        r = runs[m]
        ax = fig.add_subplot(gs[1, k])
        pc = plot_uv(ax, r["uv"], mesh["F"], r["log_area"])
        ax.plot(*r["center_2d"], marker="+", color=INK, markersize=10, mew=1.5)
        ax.set_title(m)
        ax = fig.add_subplot(gs[2, k])
        distance_axes.append(ax)
        ax.plot(s, r["d_true"] / d0, color=INK_2, label="from true 3D center")
        ax.plot(s, r["d_mapped"] / d0, color=COLORS[m], label="from mapped 3D center")
        ax.set_xlim(0, 1)
        ax.set_xlabel("3D boundary (normalized arc length)")
        ax.set_ylabel("Geodesic distance (/ mean)" if k == 0 else "")
        ax.set_title(f"{m}: mean |deviation| {100 * np.abs(r['delta']).mean():.2f} %", fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
    cb = fig.colorbar(pc, cax=fig.add_subplot(gs[1, 4]))
    cb.set_label("log(Area3D / Area2D)")
    cb.outline.set_visible(False)
    lo = min(np.min(r[k] / d0) for r in runs.values() for k in ("d_true", "d_mapped"))
    hi = max(np.max(r[k] / d0) for r in runs.values() for k in ("d_true", "d_mapped"))
    for ax in distance_axes:
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.25 * (hi - lo))
    fig.savefig(os.path.join(FIGURES, f"object_{key}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def figure_profiles():
    """Counterpart of Fig. 10: per-object geodesic deviation profiles."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharey=True)
    for ax, (key, name, _) in zip(axes.ravel(), OBJECTS):
        mesh, runs = load(key)
        for m in METHODS:  # BFF drawn wider so it stays visible where CETM coincides with it
            ax.plot(mesh["s"], 100 * runs[m]["delta"], color=COLORS[m], label=m,
                    linewidth=5 if m == "BFF" else 2)
        ax.axhline(0, color=AXIS, linewidth=1, zorder=0)
        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.set_xlabel("3D boundary (normalized arc length)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative deviation $\\delta_i$ (%)")
    lim = max(30, max(abs(a) for ax in axes.ravel() for a in ax.get_ylim()))
    axes[0, 0].set_ylim(-lim, lim)
    axes[0, 0].legend(loc="upper left", ncol=2)
    fig.suptitle("Center-to-boundary geodesic deviation profiles (Eq. 26). BFF is drawn wider so it stays visible "
                 "under overlapping lines; 0 everywhere means the mapped center is the true center.",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig10_deviation_profiles.png"), dpi=110)
    plt.close(fig)


def figure_tradeoff(rows):
    """Counterpart of Fig. 11, faceted by method (four hues cannot share one scatter safely)."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharex=True, sharey=True)
    x = {m: [float(r["mean_abs_log_area"]) for r in rows if r["method"] == m] for m in METHODS}
    y = {m: [float(r["cbgd_mean_pct"]) for r in rows if r["method"] == m] for m in METHODS}
    names = [r["object"] for r in rows if r["method"] == "LSCM"]
    for ax, m in zip(axes, METHODS):
        for other in METHODS:
            if other != m:
                ax.scatter(x[other], y[other], s=22, color=GRID, edgecolors="none", zorder=1)
        ax.scatter(x[m], y[m], s=60, color=COLORS[m], marker=MARKERS[m], edgecolors=SURFACE,
                   linewidths=1.5, zorder=3)
        for i in np.argsort(y[m])[::-1][:3]:  # label the three largest deviations only
            ax.annotate(names[i], (x[m][i], y[m][i]), xytext=(5, 3), textcoords="offset points",
                        fontsize=7, color=INK_2)
        ax.set_title(f"{m}  (mean deviation {np.mean(y[m]):.2f} %)")
        ax.set_xlabel("Mean absolute area distortion |log(A3D/A2D)|")
    axes[0].set_ylabel("Average geodesic deviation (%)")
    fig.suptitle("Per-object trade-off between area distortion and geodesic deviation "
                 "(gray: the other three methods)", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig11_area_vs_deviation.png"), dpi=110)
    plt.close(fig)


def figure_average(rows):
    """Counterpart of Fig. 12: mean (+- std over objects) geodesic deviation per method."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, m in enumerate(METHODS):
        v = np.array([float(r["cbgd_mean_pct"]) for r in rows if r["method"] == m])
        ax.bar(k, v.mean(), width=0.25, color=COLORS[m], zorder=2)
        ax.errorbar(k, v.mean(), yerr=v.std(), color=INK_2, capsize=4, linewidth=1, zorder=3)
        ax.scatter(k + 0.27 + np.linspace(-0.05, 0.05, len(v)), v, s=14, color=MUTED, edgecolors="none", zorder=3)
        ax.annotate(f"{v.mean():.2f} %\n(n={len(v)})", (k, v.mean() + v.std()), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(METHODS)), list(METHODS))
    ax.set_ylabel("Average center-to-boundary\ngeodesic deviation (%)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_title("Average geodesic deviation across all test objects\n"
                 "(bar: mean, whisker: std, gray dots: individual objects)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig12_average_deviation.png"), dpi=110)
    plt.close(fig)


def figure_spiral(key, method):
    """Counterpart of Fig. 1: spiral path designed in the UV domain and mapped onto the surface."""
    mesh, runs = load(key)
    r = runs[method]
    V, F, b = mesh["V"], mesh["F"], mesh["boundary"]
    uv100, via, found = r["uv"] * 100, r["via"], r["via_found"].astype(bool)
    fig, (a2, a3) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.2]})
    a2.add_collection(PolyCollection(uv100[F], facecolors="#f0efec", edgecolors="#c9c8c2", linewidths=0.3))
    a2.plot(*uv100[np.r_[b, b[:1]]].T, color=COLORS["BFF"], linewidth=2.5, label="Boundary")
    a2.plot(*via.T, color=INK, linewidth=1.2, label="Path")
    a2.plot(*via.T, "o", color=INK, markersize=3, label="Via points")
    a2.plot(*via[0], "o", color="#008300", markersize=9, label="Start (2D center)")
    a2.plot(*via[-1], "o", color="#e34948", markersize=9, label="End")
    a2.set_aspect("equal")
    a2.set_xlabel("u")
    a2.set_ylabel("v")
    a2.grid(False)
    a2.set_title(f"Path in the parameter domain ({method})")
    a2.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    project = render_mesh(a3, key, V, F, edges=True)
    P = project(r["path_3d"])
    a3.plot(*P.T, color=INK, linewidth=1.2)
    a3.plot(*P.T, "o", color=INK, markersize=3)
    a3.plot(*P[0], "o", color="#008300", markersize=9)
    a3.plot(*P[-1], "o", color="#e34948", markersize=9)
    a3.plot(*project(V[[int(mesh["p_true"])]]).T, "o", color=COLORS["BFF"], markersize=7, label="True 3D center")
    a3.legend(loc="lower left")
    a3.set_title(f"Path mapped to 3D ({len(r['path_3d'])} via points, step {SPIRAL_STEP} in a 100-unit frame)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, f"fig1_spiral_{key}_{method}.png"), dpi=110)
    plt.close(fig)


def figure_spacing(key, name):
    """Supplement: 3D spacing of the back-projected spiral via points for every method."""
    mesh, runs = load(key)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
    norm = Normalize(-50, 50)
    for ax, m in zip(axes, METHODS):
        r = runs[m]
        project = render_mesh(ax, key, mesh["V"], mesh["F"])
        P = project(r["path_3d"])
        keep = np.diff(np.flatnonzero(r["via_found"])) == 1  # segments between consecutive via points
        segs = np.stack([P[:-1], P[1:]], axis=1)[keep]
        ax.add_collection(LineCollection(segs, colors=INK, linewidths=4.5))  # dark casing for contrast
        ax.add_collection(LineCollection(segs, colors=DIVERGING(norm(100 * r["spacing_dev"])), linewidths=2.5))
        dev = 100 * np.abs(r["spacing_dev"])
        ax.set_title(f"{m}\nmean |step error| {dev.mean():.1f} %\n{100 * np.mean(dev > 20):.0f} % of steps off by > 20 %",
                     fontsize=10)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
    cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("3D step vs. nominal (%)")
    cb.outline.set_visible(False)
    fig.suptitle(f"{name}: spacing of the mapped spiral path (blue = compressed, red = stretched)",
                 fontsize=12, color=INK)
    fig.savefig(os.path.join(FIGURES, f"path_spacing_{key}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def figure_gallery():
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, (key, name, paper) in zip(axes.ravel(), OBJECTS):
        mesh, _ = load(key)
        render_mesh(ax, key, mesh["V"], mesh["F"], edges=len(mesh["F"]) < 20000)
        ax.set_title(f"{name}\n{len(mesh['F'])} faces" + ("" if paper == name else f" (stands in for {paper})"),
                     fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "test_meshes.png"), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------
def tables(rows):
    """Markdown tables (Table 2 counterpart and the per-object metrics) in results/tables.md."""
    objects = [name for _, name, _ in OBJECTS]

    def grid(field, fmt):
        lines = ["| Object | " + " | ".join(METHODS) + " |", "|---|" + "---:|" * len(METHODS)]
        for obj in objects:
            vals = [next(float(r[field]) for r in rows if r["object"] == obj and r["method"] == m) for m in METHODS]
            lines.append(f"| {obj} | " + " | ".join(fmt.format(v) for v in vals) + " |")
        cols = [[float(r[field]) for r in rows if r["method"] == m] for m in METHODS]
        lines.append("| **Mean +- std** | " + " | ".join(
            f"**{fmt.format(np.mean(c))} +- {fmt.format(np.std(c))}**" for c in cols) + " |")
        return "\n".join(lines)

    sections = [
        ("Average center-to-boundary geodesic deviation, mean |delta_i| (%)", "cbgd_mean_pct", "{:.2f}"),
        ("Maximum |delta_i| along the boundary (%)", "cbgd_max_pct", "{:.2f}"),
        ("Mean absolute area distortion, mean |log(A3D/A2D)|", "mean_abs_log_area", "{:.3f}"),
        ("Mean absolute angle distortion (degrees)", "mean_abs_angle_deg", "{:.2f}"),
        ("Parameterization time in seconds (median of 3 runs; Table 2 counterpart)", "time_s", "{:.3f}"),
        ("Spiral path: mean |3D step error| vs. nominal step (%)", "spacing_mean_abs_dev_pct", "{:.1f}"),
        ("Spiral path: share of 3D steps off by more than 20 % (%)", "spacing_over_20pct", "{:.1f}"),
    ]
    text = "# Results tables\n\nGenerated by `make_figures.py` from `summary.csv`.\n"
    for title, field, fmt in sections:
        text += f"\n## {title}\n\n{grid(field, fmt)}\n"
    with open(os.path.join(RESULTS, "tables.md"), "w") as f:
        f.write(text)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    rows = read_summary()
    tables(rows)
    figure_gallery()
    for key, name, _ in OBJECTS:
        figure_object(key, name)
    figure_profiles()
    figure_tradeoff(rows)
    figure_average(rows)
    figure_spiral("lilium_top", "ARAP")
    figure_spacing("bump_cap", "Bump cap")
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
