"""
Generate side-by-side visualizations (Neato vs RL) for all 99 test graphs.
Crossing edges are highlighted in red; crossing points marked with orange stars.

Usage:
    python visualize_all.py [output_dir]
    python visualize_all.py viz_output/
"""
import os
import sys
import torch
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.patheffects as pe
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xing import XingLoss

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROME_DIR  = os.path.join(BASE_DIR, 'rome')
COORD_DIR = os.path.join(BASE_DIR, 'coords_submission')
OUT_DIR   = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, 'viz_output')
os.makedirs(OUT_DIR, exist_ok=True)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _cross2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def seg_intersect(p1, p2, p3, p4, eps=1e-9):
    """
    Return intersection point of segment p1-p2 and p3-p4, or None.
    Uses parametric line intersection; excludes endpoint touches.
    """
    r   = p2 - p1
    s   = p4 - p3
    rxs = r[0]*s[1] - r[1]*s[0]
    if abs(rxs) < eps:
        return None          # parallel / collinear
    d   = p3 - p1
    t   = (d[0]*s[1] - d[1]*s[0]) / rxs
    u   = (d[0]*r[1] - d[1]*r[0]) / rxs
    tol = 1e-6
    if tol < t < 1.0 - tol and tol < u < 1.0 - tol:
        return p1 + t * r
    return None


def find_crossings(G, coords):
    """
    Iterate all non-adjacent edge pairs, detect proper intersections.
    Returns:
        crossing_edge_ids : set of edge list indices involved in ≥1 crossing
        crossing_points   : list of np.array([x, y]) intersection coordinates

    NOTE: coords rows are ordered by G.nodes() position (not node label).
    We build a label→position map so xy[label_to_pos[u]] is correct.
    """
    # Build label→position map: G.nodes() may not be sorted
    label_to_pos = {node: pos for pos, node in enumerate(G.nodes())}

    edges = list(G.edges())
    xy    = coords.numpy().astype(float)
    n     = len(edges)

    crossing_edge_ids = set()
    crossing_points   = []

    for i in range(n):
        u1, v1 = edges[i]
        p1, p2  = xy[label_to_pos[u1]], xy[label_to_pos[v1]]
        for j in range(i + 1, n):
            u2, v2 = edges[j]
            # skip adjacent edges (share a node)
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
            p3, p4 = xy[label_to_pos[u2]], xy[label_to_pos[v2]]
            pt = seg_intersect(p1, p2, p3, p4)
            if pt is not None:
                crossing_edge_ids.add(i)
                crossing_edge_ids.add(j)
                crossing_points.append(pt)

    return crossing_edge_ids, crossing_points


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_layout(G, coords, title, ax, show_crossing=True):
    xy     = coords.numpy()
    edges  = list(G.edges())

    # Build label→position map (G.nodes() order ≠ sorted node labels)
    label_to_pos = {node: pos for pos, node in enumerate(G.nodes())}

    if show_crossing:
        crossing_edge_ids, crossing_pts = find_crossings(G, coords)
    else:
        crossing_edge_ids, crossing_pts = set(), []

    # Separate normal vs crossing edges
    normal_segs   = []
    crossing_segs = []
    for i, (u, v) in enumerate(edges):
        seg = [xy[label_to_pos[u]], xy[label_to_pos[v]]]
        if i in crossing_edge_ids:
            crossing_segs.append(seg)
        else:
            normal_segs.append(seg)

    # Draw normal edges (blue-grey)
    if normal_segs:
        lc = mc.LineCollection(normal_segs, linewidths=1.0,
                               colors='steelblue', alpha=0.55, zorder=1)
        ax.add_collection(lc)

    # Draw crossing edges (red, thicker)
    if crossing_segs:
        lc2 = mc.LineCollection(crossing_segs, linewidths=1.8,
                                colors='crimson', alpha=0.85, zorder=2)
        ax.add_collection(lc2)

    # Mark crossing points with orange stars
    if crossing_pts:
        cx = [p[0] for p in crossing_pts]
        cy = [p[1] for p in crossing_pts]
        ax.scatter(cx, cy, marker='*', s=120, color='orange',
                   edgecolors='darkorange', linewidths=0.5, zorder=5)

    # Draw nodes
    ax.scatter(xy[:, 0], xy[:, 1], s=55, zorder=4,
               color='white', edgecolors='navy', linewidths=1.2)
    # Node index labels
    for i, (x, y) in enumerate(xy):
        ax.text(x, y, str(i), fontsize=5, ha='center', va='center',
                color='navy', fontweight='bold', zorder=6)

    n_cross = len(crossing_pts)
    ax.set_title(f"{title}\n{n_cross} crossing{'s' if n_cross != 1 else ''}",
                 fontsize=9, pad=4)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.axis('off')


# ── Main ──────────────────────────────────────────────────────────────────────

def load_coord(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                x, y = map(float, line.split())
                rows.append([x, y])
    return torch.tensor(rows, dtype=torch.float32)


def main():
    coord_files = sorted(
        f for f in os.listdir(COORD_DIR) if f.endswith('.coord')
    )
    print(f"Generating visualizations for {len(coord_files)} graphs → {OUT_DIR}/")

    total_spc = []
    for i, fname in enumerate(coord_files):
        stem      = fname.replace('.coord', '')
        graphml   = os.path.join(ROME_DIR, stem + '.graphml')
        coord_f   = os.path.join(COORD_DIR, fname)
        out_png   = os.path.join(OUT_DIR, stem + '.png')

        if not os.path.exists(graphml):
            print(f"  [{i+1}/{len(coord_files)}] SKIP (no graphml): {stem}")
            continue

        G = nx.read_graphml(graphml)
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        rl_coords = load_coord(coord_f)

        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
            neato_coords = torch.tensor(
                [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32
            )
            has_neato = True
        except Exception:
            has_neato = False

        xfn    = XingLoss(G, soft=False)
        rl_x   = int(xfn(rl_coords).item())
        neato_x = int(xfn(neato_coords).item()) if has_neato else None
        spc    = (neato_x - rl_x) / max(neato_x, 1) * 100 if neato_x else 0
        total_spc.append(spc)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(
            f"{stem}   N={G.number_of_nodes()}  E={G.number_of_edges()}   "
            f"SPC = {spc:+.1f}%",
            fontsize=11, fontweight='bold'
        )

        if has_neato:
            draw_layout(G, neato_coords, f"Neato baseline ({neato_x} crossings)",
                        axes[0])
        else:
            axes[0].set_visible(False)

        draw_layout(G, rl_coords, f"GATv2-PPO result ({rl_x} crossings)", axes[1])

        plt.tight_layout()
        fig.savefig(out_png, dpi=120, bbox_inches='tight')
        plt.close(fig)

        print(f"  [{i+1:3d}/{len(coord_files)}] {stem:<25}  "
              f"RL={rl_x:4d}  neato={neato_x if neato_x is not None else '?':>4}  "
              f"SPC={spc:+.1f}%  → {os.path.basename(out_png)}")

    if total_spc:
        print(f"\n{'='*55}")
        print(f"平均 SPC:  {sum(total_spc)/len(total_spc):+.1f}%")
        print(f"改善图数: {sum(1 for v in total_spc if v > 0)}/{len(total_spc)}")
        print(f"{'='*55}")
    print(f"\nDone. Images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
