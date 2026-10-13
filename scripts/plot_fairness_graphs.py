"""Draw the graphs the fairness mechanisms act on, exactly as the sweep ran them.

Two kinds of figure, written as PDF (for the paper) and SVG (for the web):

  dag_<dataset>_<role>      The hand-specified DAG every DECAF family uses
                            (`decaf`, `decaf_ctgan`, `decaf_dpgan`,
                            `decaf_diffusion`, `decaf_dpdiffusion`,
                            `decaf_tabpfn`), with the edges FTU / CF / DP cut at
                            generation time. Nothing here is re-derived: the
                            DAG is `CAUSAL_GRAPHS`, the roles are read from the
                            sweep's run rows, and the cuts come from each
                            mechanism's own `select_biased_edges`.

  <method>_<dataset>_<role> The undirected graph a marginal method selected,
                            read from the `graph_edges` rows of the sweep and
                            aggregated over seeds: one panel per mechanism, an
                            edge's width being the share of seeds that chose it.

Layout is Graphviz's (`dot` / `neato`), found on PATH or via --graphviz-bin;
`conda create -n graphviz -c conda-forge graphviz` provides it.

    python scripts/plot_fairness_graphs.py --batch facct-2026-10-09
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Polygon  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_fairness_sdg.data.causal_graphs import CAUSAL_GRAPHS, as_digraph  # noqa: E402
from causal_fairness_sdg.fairness.base import AttributeRoles  # noqa: E402
from causal_fairness_sdg.fairness.registry import FAIRNESS_MECHANISMS  # noqa: E402

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"})

NODE_FILL = {"protected": "#f28b82", "outcome": "#8ab4f8", "admissible": "#fde293", "other": "#e8eaed"}
NODE_LABEL = {
    "protected": "protected", "outcome": "outcome", "admissible": "admissible", "other": "other",
}
#: Mechanism -> (colour of its cut mark, the letter drawn beside the mark so
#: the three are told apart without colour).
MECHANISMS = {"ftu": ("#1e9e3f", "F"), "cf": ("#f2b600", "C"), "dp": ("#d93025", "D")}
MECHANISM_NAME = {"none": "No mechanism", "ftu": "FTU", "cf": "CF", "dp": "DP"}
INK, EDGE, GHOST, ADDED = "#202124", "#5f6368", "#bdc1c6", "#6a3d9a"

#: Datasets that share one DAG and one set of role splits are drawn once.
DAG_FIGURES = {
    "acsincome": "acsincome_ca", "acspubcov": "acspubcov_ca",
    "cdc_diabetes": "cdc_diabetes", "sbo": "sbo", "snake": "snake",
}


# --------------------------------------------------------------------- data

def _runs(batch_dir: Path, dataset: str, method: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(str(batch_dir / "db" / f"{dataset}-{method}-s*.db"))):
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        runs = pd.read_sql("SELECT * FROM runs WHERE status = 'done'", conn)
        runs["shard"] = Path(path).stem
        edges = pd.read_sql("SELECT run_id, node_a, node_b FROM graph_edges", conn)
        runs["edges"] = runs.run_id.map(
            edges.groupby("run_id").apply(
                lambda g: [tuple(e) for e in g[["node_a", "node_b"]].to_numpy()]
            )
        )
        frames.append(runs)
        conn.close()
    if not frames:
        raise SystemExit(f"no finished {method} runs for {dataset} under {batch_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["role"] = out.extra_params.map(lambda s: json.loads(s)["role_config"])
    return out


def _roles(runs: pd.DataFrame) -> Dict[str, AttributeRoles]:
    """Role split name -> the roles the runs were actually given."""
    out = {}
    for role, rows in runs.groupby("role", sort=False):
        row = rows.iloc[0]
        out[role] = AttributeRoles.create(
            json.loads(row.protected_attrs), json.loads(row.admissible_attrs or "[]"),
            [row.outcome_attr],
        )
    return out


def _kind(node: str, roles: AttributeRoles) -> str:
    for kind in ("protected", "outcome", "admissible"):
        if node in getattr(roles, kind):
            return kind
    return "other"


# ------------------------------------------------------------------- layout

def _graphviz(engine: str, bin_dir: str | None) -> str:
    found = shutil.which(engine, path=bin_dir) or shutil.which(engine)
    if not found:
        raise SystemExit(f"Graphviz `{engine}` not found; pass --graphviz-bin")
    return found


def _layout(nodes, edges, directed: bool, bin_dir: str | None):
    """Node boxes and edge splines from Graphviz, in inches."""
    arrow = "->" if directed else "--"
    lines = [
        ("digraph" if directed else "graph") + " G {",
        'node [shape=box, fontname="DejaVu Sans", fontsize=10.5, margin="0.09,0.05", height=0.3];',
        "edge [arrowsize=0.6];",
    ]
    if directed:
        lines.append("rankdir=LR; ranksep=0.75; nodesep=0.16;")
    else:
        lines.append("overlap=false; splines=true; sep=\"+6\"; start=7;")
    lines += [f'"{n}";' for n in nodes]
    for a, b in edges:
        extra = " [len=0.8]" if not directed else ""
        lines.append(f'"{a}" {arrow} "{b}"{extra};')
    lines.append("}")
    plain = subprocess.run(
        [_graphviz("dot" if directed else "neato", bin_dir), "-Tplain"],
        input="\n".join(lines), capture_output=True, text=True, check=True,
    ).stdout
    boxes, splines, size = {}, {}, (0.0, 0.0)
    for line in plain.splitlines():
        parts = shlex.split(line)
        if parts[0] == "graph":
            size = (float(parts[2]), float(parts[3]))
        elif parts[0] == "node":
            boxes[parts[1]] = tuple(float(v) for v in parts[2:6])
        elif parts[0] == "edge":
            n = int(parts[3])
            pts = np.array(parts[4:4 + 2 * n], dtype=float).reshape(n, 2)
            splines[(parts[1], parts[2])] = pts
    return boxes, splines, size


def _sample(ctrl: np.ndarray, per_segment: int = 24) -> np.ndarray:
    """Polyline along Graphviz's piecewise cubic Bezier."""
    t = np.linspace(0, 1, per_segment)[:, None]
    out = []
    for i in range(0, len(ctrl) - 1, 3):
        p0, p1, p2, p3 = ctrl[i:i + 4]
        out.append((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)
    return np.concatenate(out)


def _along(line: np.ndarray, fraction: float, shift: float = 0.0):
    """Point and unit tangent at `fraction` of the arc length, plus `shift` inches."""
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.clip(fraction * cum[-1] + shift, 0.0, cum[-1])
    i = min(np.searchsorted(cum, target, side="right") - 1, len(seg) - 1)
    tangent = (line[i + 1] - line[i]) / max(seg[i], 1e-9)
    return line[i] + tangent * (target - cum[i]), tangent


# ------------------------------------------------------------------ drawing

def _draw_nodes(ax, boxes, roles):
    for node, (x, y, w, h) in boxes.items():
        ax.add_patch(FancyBboxPatch(
            (x - w / 2 + 0.03, y - h / 2 + 0.03), w - 0.06, h - 0.06,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            facecolor=NODE_FILL[_kind(node, roles)], edgecolor=INK, linewidth=0.7, zorder=3,
        ))
        ax.text(x, y, node, ha="center", va="center", fontsize=7.6, color=INK, zorder=4)


def _draw_cross(ax, at, tangent, colour, letter):
    r = 0.062
    normal = np.array([-tangent[1], tangent[0]])
    for d in (tangent + normal, tangent - normal):
        d = d / np.linalg.norm(d) * r
        ax.plot(
            [at[0] - d[0], at[0] + d[0]], [at[1] - d[1], at[1] + d[1]], color=colour,
            linewidth=2.3, solid_capstyle="round", zorder=6,
            path_effects=[pe.Stroke(linewidth=3.9, foreground=INK), pe.Normal()],
        )
    if normal[1] < 0:
        normal = -normal
    ax.text(
        *(at + normal * 0.115), letter, fontsize=5.4, fontweight="bold", color=INK,
        ha="center", va="center", zorder=7,
        path_effects=[pe.Stroke(linewidth=1.6, foreground="white"), pe.Normal()],
    )


def _frame(ax, size, pad=0.12):
    ax.set_xlim(-pad, size[0] + pad)
    ax.set_ylim(-pad, size[1] + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def _node_legend(roles) -> List[Patch]:
    return [
        Patch(facecolor=NODE_FILL[k], edgecolor=INK, linewidth=0.7, label=NODE_LABEL[k])
        for k in ("protected", "admissible", "outcome", "other")
    ]


def _cross_handle(mechanism: str, label: str) -> Line2D:
    colour, letter = MECHANISMS[mechanism]
    return Line2D(
        [], [], linestyle="none", marker="x", markersize=7, markeredgewidth=2.3,
        markeredgecolor=colour, label=f"{label} ({letter})",
        path_effects=[pe.Stroke(linewidth=3.9, foreground=INK), pe.Normal()],
    )


def _save(fig, out_dir: Path, stem: str):
    for ext in ("pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    # Web hosts refuse SVG that carries a DTD reference; nothing needs it.
    svg = out_dir / f"{stem}.svg"
    svg.write_text(re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg.read_text(), count=1))


def dag_cuts(dag, roles) -> Dict[str, List[Tuple[str, str]]]:
    """Mechanism -> [(parent, child)] edges whose parent is permuted when the
    child is sampled; the same call `_BaseDECAFVariant._biased_edges` makes."""
    return {
        name: sorted(
            (parent, child)
            for child, parents in FAIRNESS_MECHANISMS[name]().select_biased_edges(dag, roles).items()
            for parent in parents
        )
        for name in MECHANISMS
    }


def draw_dag(dataset, role, roles, columns, out_dir, stem, bin_dir) -> dict:
    dag = as_digraph(CAUSAL_GRAPHS[dataset], columns)
    cuts = dag_cuts(dag, roles)
    boxes, splines, size = _layout(list(dag.nodes), list(dag.edges), True, bin_dir)
    fig, ax = plt.subplots(figsize=(size[0] + 0.3, size[1] + 0.75))
    for (a, b), ctrl in splines.items():
        line = _sample(ctrl)
        ax.plot(line[:, 0], line[:, 1], color=EDGE, linewidth=0.75, zorder=1)
        tip, tangent = line[-1], line[-1] - line[-2]
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        ax.add_patch(Polygon(
            [tip + tangent * 0.1, tip + normal * 0.032, tip - normal * 0.032],
            closed=True, facecolor=EDGE, edgecolor=EDGE, linewidth=0.3, zorder=2,
        ))
        marks = [m for m in MECHANISMS if (a, b) in cuts[m]]
        for k, mechanism in enumerate(marks):
            at, t = _along(line, 0.5, shift=(k - (len(marks) - 1) / 2) * 0.2)
            _draw_cross(ax, at, t, *MECHANISMS[mechanism])
    _draw_nodes(ax, boxes, roles)
    _frame(ax, size)
    handles = _node_legend(roles) + [
        _cross_handle("ftu", "cut by FTU"), _cross_handle("cf", "cut by CF"),
        _cross_handle("dp", "cut by DP"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=7.5,
               handlelength=1.2, columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
    _save(fig, out_dir, stem)
    return {
        "nodes": dag.number_of_nodes(), "edges": dag.number_of_edges(),
        "cuts": {m: [f"{a} → {b}" for a, b in cuts[m]] for m in MECHANISMS},
    }


def draw_marginal(method, dataset, role, roles, runs, epsilon, out_dir, stem, bin_dir) -> dict:
    """One panel per mechanism; edge width is the share of seeds choosing it."""
    rows = runs[(runs.role == role) & (runs.epsilon == epsilon)]
    seeds = sorted(rows.seed.unique())
    share: Dict[str, Dict[Tuple[str, str], float]] = {}
    for mechanism, group in rows.groupby("fairness_mechanism"):
        counts = defaultdict(int)
        for edges in group.edges:
            for edge in {tuple(sorted(e)) for e in edges}:
                counts[edge] += 1
        share[mechanism] = {e: c / group.seed.nunique() for e, c in counts.items()}
    union = sorted({e for s in share.values() for e in s})
    nodes = [c for c in runs.attrs["columns"]]
    boxes, splines, size = _layout(nodes, union, False, bin_dir)

    fig, axes = plt.subplots(2, 2, figsize=(2 * size[0] + 0.5, 2 * size[1] + 1.2))
    summary = {}
    for ax, mechanism in zip(axes.ravel(), ["none", "ftu", "cf", "dp"]):
        here, base = share.get(mechanism, {}), share["none"]
        dropped, added = [], []
        for edge in union:
            line = _sample(splines[edge]) if edge in splines else _sample(splines[edge[::-1]])
            f, f0 = here.get(edge, 0.0), base.get(edge, 0.0)
            if mechanism != "none" and f0 > 0 and f == 0:
                ax.plot(line[:, 0], line[:, 1], color=GHOST, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
                _draw_cross(ax, *_along(line, 0.5), *MECHANISMS[mechanism])
                dropped.append(" – ".join(edge))
                continue
            if f == 0:
                continue
            is_added = mechanism != "none" and f0 == 0
            if is_added:
                added.append(" – ".join(edge))
            ax.plot(line[:, 0], line[:, 1], color=ADDED if is_added else EDGE,
                    linewidth=0.5 + 1.7 * f, zorder=1, solid_capstyle="round")
            if f < 1:
                at, _ = _along(line, 0.5)
                ax.text(*at, f"{round(f * len(seeds))}/{len(seeds)}", fontsize=5.2, color=INK,
                        ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"))
        _draw_nodes(ax, boxes, roles)
        _frame(ax, size)
        ax.set_title(MECHANISM_NAME[mechanism], fontsize=10, color=INK, loc="left", pad=3)
        summary[mechanism] = {"dropped": dropped, "added": added}
    handles = _node_legend(roles) + [
        Line2D([], [], color=EDGE, linewidth=2.2, label=f"chosen in all {len(seeds)} seeds"),
        Line2D([], [], color=EDGE, linewidth=0.9, label="chosen in some (count shown)"),
        Line2D([], [], color=GHOST, linewidth=1.0, linestyle=(0, (3, 2)),
               marker="x", markeredgecolor=INK, markersize=6,
               label="chosen without the mechanism, never with it (F / C / D)"),
        Line2D([], [], color=ADDED, linewidth=2.0, label="chosen only with the mechanism"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               handlelength=1.8, columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.03, hspace=0.08, left=0.01, right=0.99, top=0.97, bottom=0.07)
    _save(fig, out_dir, stem)
    return {"seeds": len(seeds), "epsilon": epsilon, "changes": summary}


# --------------------------------------------------------------------- main

def _columns(batch_dir: Path, dataset: str) -> List[str]:
    pattern = str(batch_dir / "synth" / f"{dataset}-*-s0" / "run_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no saved synthetic table for {dataset} ({pattern})")
    return list(pd.read_parquet(files[0]).columns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", default="facct-2026-10-09")
    parser.add_argument("--out", default=str(REPO_ROOT / "manuscript" / "figs" / "graphs"))
    parser.add_argument("--marginal-methods", nargs="*", default=["mst"])
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="privacy budget whose marginal-method graphs are drawn")
    parser.add_argument("--dag-method", default="decaf_tabpfn",
                        help="DAG family whose run rows supply the role splits")
    parser.add_argument("--graphviz-bin", default=None)
    args = parser.parse_args()

    batch_dir = REPO_ROOT / "sweeps" / args.batch
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads((batch_dir / "sweep.json").read_text())
    index = {"batch": args.batch, "epsilon": args.epsilon, "dag": [], "marginal": []}

    for label, dataset in DAG_FIGURES.items():
        columns = _columns(batch_dir, dataset)
        for role, roles in _roles(_runs(batch_dir, dataset, args.dag_method)).items():
            stem = f"dag_{label}_{role}"
            info = draw_dag(dataset, role, roles, columns, out_dir, stem, args.graphviz_bin)
            index["dag"].append({
                "dataset": label, "role": role, "file": stem,
                "protected": sorted(roles.protected), "admissible": sorted(roles.admissible),
                "outcome": sorted(roles.outcome), **info,
            })
            print("wrote", stem)

    for method in args.marginal_methods:
        for dataset in spec["datasets"]:
            runs = _runs(batch_dir, dataset, method)
            runs.attrs["columns"] = _columns(batch_dir, dataset)
            for role, roles in _roles(runs).items():
                stem = f"{method}_{dataset}_{role}_eps{args.epsilon:g}"
                info = draw_marginal(method, dataset, role, roles, runs, args.epsilon,
                                     out_dir, stem, args.graphviz_bin)
                index["marginal"].append({
                    "method": method, "dataset": dataset, "role": role, "file": stem,
                    "protected": sorted(roles.protected), "admissible": sorted(roles.admissible),
                    "outcome": sorted(roles.outcome), **info,
                })
                print("wrote", stem)

    (out_dir / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
