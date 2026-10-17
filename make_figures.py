"""Figures and tables from the saved results.

Counterparts of the paper's Figs. 1-12 and Table 2 for its four methods, and
extension figures (prefix ext_) comparing all ten methods.

Usage:  python make_figures.py      (after run_experiments.py)
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from flattening import FAMILY, METHODS, PAPER_METHODS  # noqa: E402
from run_experiments import DATA, OBJECTS, RESULTS, SPIRAL_STEP  # noqa: E402

FIGURES = os.path.join(RESULTS, "figures")

# Paper figures: categorical slots 1-4 of the validated reference palette, one color per method.
COLORS = {"LSCM": "#2a78d6", "BFF": "#eb6834", "CETM": "#1baf7a", "ARAP": "#eda100"}
MARKERS = {"LSCM": "o", "BFF": "s", "CETM": "D", "ARAP": "^"}
# Extension figures: ten methods are too many hues, so color only tells the paper's methods
# from the added ones (slots 1-2, validated all-pairs) and every method is labeled directly.
GROUP = {m: "paper" if m in PAPER_METHODS else "added" for m in METHODS}
GROUP_COLORS = {"paper": "#2a78d6", "added": "#eb6834"}
GROUP_LABELS = {"paper": "compared in the paper", "added": "added by the extension"}
# Sequential blue ramp (steps 100-700) for magnitudes.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
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


def per_object(rows, field, methods=None):
    """{method: values over the objects, in OBJECTS order} (all methods by default)."""
    return {m: np.array([float(next(r[field] for r in rows if r["object"] == name and r["method"] == m))
                         for _, name, _ in OBJECTS]) for m in (METHODS if methods is None else methods)}


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
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=m) for m in PAPER_METHODS]
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
    for k, m in enumerate(PAPER_METHODS):
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
    paper_runs = [runs[m] for m in PAPER_METHODS]
    lo = min(np.min(r[k] / d0) for r in paper_runs for k in ("d_true", "d_mapped"))
    hi = max(np.max(r[k] / d0) for r in paper_runs for k in ("d_true", "d_mapped"))
    for ax in distance_axes:
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.25 * (hi - lo))
    fig.savefig(os.path.join(FIGURES, f"object_{key}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def figure_profiles():
    """Counterpart of Fig. 10: per-object geodesic deviation profiles."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharey=True)
    for ax, (key, name, _) in zip(axes.ravel(), OBJECTS):
        mesh, runs = load(key)
        for m in PAPER_METHODS:  # BFF drawn wider so it stays visible where CETM coincides with it
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
    x = per_object(rows, "mean_abs_log_area", PAPER_METHODS)
    y = per_object(rows, "cbgd_mean_pct", PAPER_METHODS)
    names = [name for _, name, _ in OBJECTS]
    for ax, m in zip(axes, PAPER_METHODS):
        for other in PAPER_METHODS:
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
    for k, m in enumerate(PAPER_METHODS):
        v = per_object(rows, "cbgd_mean_pct", [m])[m]
        ax.bar(k, v.mean(), width=0.25, color=COLORS[m], zorder=2)
        ax.errorbar(k, v.mean(), yerr=v.std(), color=INK_2, capsize=4, linewidth=1, zorder=3)
        ax.scatter(k + 0.27 + np.linspace(-0.05, 0.05, len(v)), v, s=14, color=MUTED, edgecolors="none", zorder=3)
        ax.annotate(f"{v.mean():.2f} %\n(n={len(v)})", (k, v.mean() + v.std()), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(PAPER_METHODS)), list(PAPER_METHODS))
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


def figure_spacing(key, name, methods=tuple(PAPER_METHODS), filename=None):
    """Supplement: 3D spacing of the back-projected spiral via points for every method."""
    mesh, runs = load(key)
    cols = len(methods) if len(methods) <= 6 else int(np.ceil(len(methods) / 2))
    rows_ = int(np.ceil(len(methods) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(17 / 4 * cols, (4.8 if rows_ == 1 else 3.6) * rows_),
                             squeeze=False)
    for ax in axes.ravel()[len(methods):]:
        ax.axis("off")
    norm = Normalize(-50, 50)
    for ax, m in zip(axes.ravel(), methods):
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
    fig.savefig(os.path.join(FIGURES, filename or f"path_spacing_{key}.png"), dpi=110, bbox_inches="tight")
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
# Extension figures (all ten methods)
# ----------------------------------------------------------------------------------------
def group_legend(ax, **kw):
    ax.legend(handles=[Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g]) for g in GROUP_COLORS], **kw)


def label_points(ax, x, y, names, min_sep=0.04):
    """Direct labels. Points closer than min_sep (in axes fractions) share one label; each label goes
    to the first of several positions around its point that overlaps no other label or point."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = ax.transAxes.inverted().transform(ax.transData.transform(np.c_[x, y]))
    groups = []
    for i in np.argsort(x):
        for g in groups:
            if np.min(np.linalg.norm(pts[g] - pts[i], axis=1)) < min_sep:
                g.append(i)
                break
        else:
            groups.append([i])
    frame = ax.get_window_extent(renderer)
    taken = [Bbox.from_bounds(px - 5, py - 5, 10, 10) for px, py in ax.transData.transform(np.c_[x, y])]
    spots = [(7, 0, "left", "center"), (-7, 0, "right", "center"), (0, 7, "center", "bottom"),
             (0, -7, "center", "top"), (7, 8, "left", "bottom"), (7, -8, "left", "top"),
             (-7, 8, "right", "bottom"), (-7, -8, "right", "top")]
    for g in groups:
        k = max(g, key=lambda i: x[i])
        text = ", ".join(names[i] for i in sorted(g, key=lambda i: -y[i]))
        for n_try, (dx, dy, ha, va) in enumerate(spots):
            label = ax.annotate(text, (x[k], y[k]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                                fontsize=8, color=INK_2)
            box = label.get_window_extent(renderer)
            inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
            if inside and not any(box.overlaps(t) for t in taken) or n_try == len(spots) - 1:
                taken.append(box)
                break
            label.remove()


def figure_ext_average(rows):
    """All methods: mean (+- std over objects) center-to-boundary geodesic deviation."""
    v = per_object(rows, "cbgd_mean_pct")
    order = sorted(METHODS, key=lambda m: -v[m].mean())  # smallest deviation at the top
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    for k, m in enumerate(order):
        mean, std = v[m].mean(), v[m].std()
        ax.barh(k, mean, height=0.42, color=GROUP_COLORS[GROUP[m]], zorder=2)
        ax.errorbar(mean, k, xerr=std, color=INK_2, capsize=3, linewidth=1, zorder=3)
        ax.scatter(v[m], np.full(len(v[m]), k + 0.34), s=12, color=MUTED, edgecolors="none", zorder=3)
        ax.annotate(f"{mean:.2f} %", (mean + std, k), xytext=(5, -3), textcoords="offset points", fontsize=8,
                    color=INK)
    ax.set_yticks(range(len(order)), [f"{m} ({FAMILY[m]})" for m in order])
    ax.set_xlabel("Average center-to-boundary geodesic deviation (%)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(np.max(v[m]) for m in METHODS) * 1.08)
    group_legend(ax, loc="lower right")
    ax.set_title("Average geodesic deviation over the 8 objects\n(bar: mean, whisker: std, "
                 "gray dots: individual objects)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "ext_average_deviation.png"), dpi=110)
    plt.close(fig)


def figure_ext_tradeoff(rows):
    """Method means: area vs. angle distortion, and area distortion vs. geodesic deviation."""
    names = list(METHODS)
    mean = {f: np.array([v.mean() for v in per_object(rows, f).values()])
            for f in ("mean_abs_log_area", "mean_abs_angle_deg", "cbgd_mean_pct")}
    x = mean["mean_abs_log_area"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    panels = [(axes[0], mean["mean_abs_angle_deg"], "Mean angle distortion (degrees, log scale)", True,
               "Area vs. angle distortion"),
              (axes[1], mean["cbgd_mean_pct"], "Average geodesic deviation, CBGD (%)", False,
               "Area distortion vs. geodesic deviation")]
    for ax, y, ylabel, log_y, title in panels:
        ax.scatter(x, y, s=60, c=[GROUP_COLORS[GROUP[m]] for m in names], edgecolors=SURFACE, linewidths=1.5,
                   zorder=3)
        if log_y:
            ax.set_yscale("log")
            ax.set_ylim(y.min() / 1.8, y.max() * 1.8)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.set_ylim(0, y.max() * 1.15)
        ax.set_xlim(0, x.max() * 1.3)
        ax.set_xlabel("Mean area distortion |log(A3D/A2D)|")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} (method means over the 8 objects)", fontsize=10)
        label_points(ax, x, y, names)
    group_legend(axes[1], loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "ext_tradeoff.png"), dpi=110)
    plt.close(fig)


def figure_ext_heatmap(rows):
    """CBGD of every method on every object."""
    v = per_object(rows, "cbgd_mean_pct")
    M = np.array([v[m] for m in METHODS]).T
    M = np.vstack([M, M.mean(axis=0)])
    labels = [name for _, name, _ in OBJECTS] + ["Mean"]
    fig, ax = plt.subplots(figsize=(13, 6))
    norm = PowerNorm(0.5, vmin=0, vmax=M.max())
    im = ax.imshow(M, cmap=SEQUENTIAL, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if norm(M[i, j]) > 0.55 else INK)
    ax.set_xticks(range(len(METHODS)), list(METHODS))
    ax.set_yticks(range(len(labels)), labels)
    ax.axhline(len(OBJECTS) - 0.5, color="white", linewidth=4)
    ax.axvline(len(PAPER_METHODS) - 0.5, color="white", linewidth=4)
    ax.grid(False)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("CBGD (%), square-root color scale")
    cb.outline.set_visible(False)
    ax.set_title("Center-to-boundary geodesic deviation (%) per object; left of the gap: the paper's methods",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "ext_cbgd_heatmap.png"), dpi=110)
    plt.close(fig)


def figure_ext_uv_grid():
    """Area distortion maps of every method (columns) on every object (rows)."""
    fig, axes = plt.subplots(len(OBJECTS), len(METHODS), figsize=(1.6 * len(METHODS), 1.6 * len(OBJECTS) + 0.6))
    for i, (key, name, _) in enumerate(OBJECTS):
        mesh, runs = load(key)
        for j, m in enumerate(METHODS):
            ax = axes[i, j]
            pc = plot_uv(ax, runs[m]["uv"], mesh["F"], runs[m]["log_area"])
            ax.plot(*runs[m]["center_2d"], marker="+", color=INK, markersize=6, mew=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel(name if j == 0 else "", fontsize=9)
            for side in ax.spines.values():
                side.set_visible(False)
            if i == 0:
                ax.set_title(m, fontsize=10)
    cb = fig.colorbar(pc, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("log(Area3D / Area2D)")
    cb.outline.set_visible(False)
    fig.savefig(os.path.join(FIGURES, "ext_uv_grid.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)


def figure_ext_object(key, name, rows):
    """One object, all methods: area distortion maps with CBGD and validity in the titles."""
    mesh, runs = load(key)
    fig, axes = plt.subplots(3, 4, figsize=(15, 12), gridspec_kw={"hspace": 0.42, "wspace": 0.25})
    for ax in axes.ravel()[1 + len(METHODS):]:
        ax.axis("off")
    render_mesh(axes[0, 0], key, mesh["V"], mesh["F"], edges=len(mesh["F"]) < 20000)
    axes[0, 0].set_title(f"{name} ({len(mesh['F'])} faces)", fontsize=11)
    for ax, m in zip(axes.ravel()[1:], METHODS):
        r = runs[m]
        row = next(x for x in rows if x["object"] == name and x["method"] == m)
        pc = plot_uv(ax, r["uv"], mesh["F"], r["log_area"])
        ax.plot(*r["center_2d"], marker="+", color=INK, markersize=10, mew=1.5)
        issues = [f"{row['flipped_faces']} flipped"] if int(row["flipped_faces"]) else []
        issues += [f"{row['boundary_self_intersections']} boundary crossings"] if int(
            row["boundary_self_intersections"]) else []
        ax.set_title(f"{m}: CBGD {float(row['cbgd_mean_pct']):.2f} %" + ("\n" + ", ".join(issues) if issues else ""),
                     fontsize=10)
    cb = fig.colorbar(pc, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("log(Area3D / Area2D)")
    cb.outline.set_visible(False)
    fig.savefig(os.path.join(FIGURES, f"ext_object_{key}.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)


def figure_ext_timing(rows):
    """Flattening time of every method on every object (log scale)."""
    t = per_object(rows, "time_s")
    order = sorted(METHODS, key=lambda m: -np.median(t[m]))  # fastest at the top
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for k, m in enumerate(order):
        ax.scatter(t[m], np.full(len(t[m]), k), s=18, color=MUTED, edgecolors="none", zorder=2)
        med = np.median(t[m])
        ax.plot([med, med], [k - 0.32, k + 0.32], color=GROUP_COLORS[GROUP[m]], linewidth=3, zorder=3)
        ax.annotate(f"median {med:.3g} s", (t[m].max(), k), xytext=(7, -3), textcoords="offset points", fontsize=8,
                    color=INK)
    ax.set_xscale("log")
    ax.set_xlim(min(np.min(v) for v in t.values()) / 2, max(np.max(v) for v in t.values()) * 12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks(range(len(order)), order)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Flattening time (s, log scale)")
    group_legend(ax, loc="upper right")
    ax.set_title("Time per object (gray dots) and median over the 8 objects (bar)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "ext_timing.png"), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------
def tables(rows):
    """Markdown tables (Table 2 counterpart and the per-object metrics) in results/tables.md."""
    objects = [name for _, name, _ in OBJECTS]
    COUNTS = {"flipped_faces", "boundary_self_intersections", "spiral_unmapped"}

    def grid(field, fmt):
        lines = ["| Object | " + " | ".join(METHODS) + " |", "|---|" + "---:|" * len(METHODS)]
        for obj in objects:
            vals = [next(float(r[field]) for r in rows if r["object"] == obj and r["method"] == m) for m in METHODS]
            lines.append(f"| {obj} | " + " | ".join(fmt.format(v) for v in vals) + " |")
        cols = [[float(r[field]) for r in rows if r["method"] == m] for m in METHODS]
        if field in COUNTS:  # for validity counts, the number of affected objects says more than a mean
            lines.append("| **Objects affected** | " + " | ".join(
                f"**{int(np.sum(np.array(c) > 0))}**" for c in cols) + " |")
        else:
            lines.append("| **Mean +- std** | " + " | ".join(
                f"**{fmt.format(np.mean(c))} +- {fmt.format(np.std(c))}**" for c in cols) + " |")
        return "\n".join(lines)

    sections = [
        ("Average center-to-boundary geodesic deviation, mean |delta_i| (%)", "cbgd_mean_pct", "{:.2f}"),
        ("Maximum |delta_i| along the boundary (%)", "cbgd_max_pct", "{:.2f}"),
        ("Mean absolute area distortion, mean |log(A3D/A2D)|", "mean_abs_log_area", "{:.3f}"),
        ("Mean absolute angle distortion (degrees)", "mean_abs_angle_deg", "{:.2f}"),
        ("Parameterization time in seconds (median of 3 runs, one run for methods slower than 60 s; "
         "Table 2 counterpart)", "time_s", "{:.3f}"),
        ("Spiral path: mean |3D step error| vs. nominal step (%)", "spacing_mean_abs_dev_pct", "{:.1f}"),
        ("Spiral path: share of 3D steps off by more than 20 % (%)", "spacing_over_20pct", "{:.1f}"),
        ("Validity: flipped or degenerate UV triangles", "flipped_faces", "{:.0f}"),
        ("Validity: crossing pairs of boundary edges (global overlaps)", "boundary_self_intersections", "{:.0f}"),
        ("Validity: spiral via points outside the UV mesh", "spiral_unmapped", "{:.0f}"),
    ]
    text = ("# Results tables\n\nGenerated by `make_figures.py` from `summary.csv`. LSCM, BFF, CETM and ARAP are "
            "the paper's methods; the other six were added by the extension.\n")
    for title, field, fmt in sections:
        text += f"\n## {title}\n\n{grid(field, fmt)}\n"
    text += f"\n## Does distortion predict the geodesic deviation?\n\n{correlations(rows)}\n"
    text += f"\n## Objects on which each method is best (ties count for every tied method)\n\n{wins(rows)}\n"
    with open(os.path.join(RESULTS, "tables.md"), "w") as f:
        f.write(text)


def wins(rows):
    """Per method: objects with the lowest CBGD and the lowest path step error (ties to 0.01 count for all)."""
    lines = ["| Measure | " + " | ".join(METHODS) + " |", "|---|" + "---:|" * len(METHODS)]
    for title, field in [("Lowest CBGD", "cbgd_mean_pct"), ("Lowest path step error", "spacing_mean_abs_dev_pct")]:
        v = per_object(rows, field)
        best = np.min(np.array([v[m] for m in METHODS]), axis=0)
        lines.append(f"| {title} | " + " | ".join(str(int(np.sum(v[m] <= best + 0.005))) for m in METHODS) + " |")
    return "\n".join(lines)


def correlations(rows):
    """Spearman rank correlation of each distortion measure with CBGD, over the methods for each object."""
    cbgd = per_object(rows, "cbgd_mean_pct")
    lines = ["| Measure | median rho over objects | objects with rho > 0 |", "|---|---:|---:|"]
    for title, field in [("Area distortion", "mean_abs_log_area"), ("Angle distortion", "mean_abs_angle_deg"),
                         ("Path step error", "spacing_mean_abs_dev_pct")]:
        x = per_object(rows, field)
        rho = []
        for i in range(len(OBJECTS)):
            a = np.array([x[m][i] for m in METHODS])
            b = np.array([cbgd[m][i] for m in METHODS])
            if np.ptp(b) > 0:  # skip objects on which every method has the same CBGD
                rho.append(spearmanr(a, b)[0])
        lines.append(f"| {title} | {np.median(rho):.2f} | {np.sum(np.array(rho) > 0)} of {len(rho)} |")
    return ("For each object, the Spearman rank correlation across all methods between the distortion measure "
            "and CBGD; objects on which all methods have the same CBGD are skipped.\n\n" + "\n".join(lines))


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
    figure_ext_average(rows)
    figure_ext_tradeoff(rows)
    figure_ext_heatmap(rows)
    figure_ext_timing(rows)
    figure_ext_uv_grid()
    for key, name, _ in OBJECTS:
        figure_ext_object(key, name, rows)
    figure_spacing("bump_cap", "Bump cap", methods=tuple(METHODS), filename="ext_path_spacing_bump_cap.png")
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
