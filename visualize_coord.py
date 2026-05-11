"""
Read a .coord file + its .graphml, draw the graph, and count crossings.

Usage:
    python visualize_coord.py coords_submission/grafo10000.38.coord
    python visualize_coord.py coords_submission/grafo10000.38.coord --no-show  # just print stats
"""
import os
import sys
import torch
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xing import XingLoss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROME_DIR = os.path.join(BASE_DIR, 'rome')


def load_coord(coord_path):
    coords = []
    with open(coord_path) as f:
        for line in f:
            line = line.strip()
            if line:
                x, y = map(float, line.split())
                coords.append([x, y])
    return torch.tensor(coords, dtype=torch.float32)


def draw_graph(G, coords, title="", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    xy = coords.numpy()
    # Draw edges
    segments = []
    for u, v in G.edges():
        segments.append([xy[u], xy[v]])
    lc = mc.LineCollection(segments, linewidths=0.8, colors='steelblue', alpha=0.6)
    ax.add_collection(lc)
    # Draw nodes
    ax.scatter(xy[:, 0], xy[:, 1], s=40, zorder=3, color='tomato', edgecolors='black', linewidths=0.5)
    # Node labels
    for i, (x, y) in enumerate(xy):
        ax.annotate(str(i), (x, y), fontsize=6, ha='center', va='center', color='white', fontweight='bold')
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    return ax


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_coord.py <path/to/graph.coord> [--no-show]")
        sys.exit(1)

    coord_path = sys.argv[1]
    show = '--no-show' not in sys.argv

    # Infer graph name from coord filename (e.g. grafo10000.38.coord → grafo10000.38.graphml)
    basename = os.path.basename(coord_path).replace('.coord', '.graphml')
    graphml_path = os.path.join(ROME_DIR, basename)
    if not os.path.exists(graphml_path):
        print(f"ERROR: could not find {graphml_path}")
        sys.exit(1)

    G = nx.read_graphml(graphml_path)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    coords = load_coord(coord_path)

    xfn = XingLoss(G, soft=False)
    xings = int(xfn(coords).item())

    # Also compute neato baseline for comparison
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
        neato_coords = torch.tensor(
            [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32
        )
        neato_xings = int(xfn(neato_coords).item())
        spc = (neato_xings - xings) / max(neato_xings, 1) * 100
        print(f"Graph:        {basename.replace('.graphml','')}")
        print(f"Nodes:        {G.number_of_nodes()}")
        print(f"Edges:        {G.number_of_edges()}")
        print(f"Crossings:    {xings}")
        print(f"Neato:        {neato_xings}")
        print(f"SPC:          {spc:+.1f}%")
    except Exception:
        neato_coords = None
        neato_xings = None
        print(f"Graph:     {basename.replace('.graphml','')}")
        print(f"Crossings: {xings}")

    if show:
        if neato_coords is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            draw_graph(G, neato_coords, f"Neato  ({neato_xings} crossings)", ax=ax1)
            draw_graph(G, coords,       f"RL     ({xings} crossings)  SPC={spc:+.1f}%", ax=ax2)
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            draw_graph(G, coords, f"{basename}  ({xings} crossings)", ax=ax)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
