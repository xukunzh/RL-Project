"""
Evaluate the current coords_submission/ files against neato baseline.
Usage:
    python eval_submission.py
"""
import os
import sys
import torch
import networkx as nx

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xing import XingLoss

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROME_DIR  = os.path.join(BASE_DIR, 'rome')
COORD_DIR = os.path.join(BASE_DIR, 'coords_submission')


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
    coord_files = sorted(f for f in os.listdir(COORD_DIR) if f.endswith('.coord'))
    print(f"Found {len(coord_files)} coord files.\n")

    header = f"{'Graph':<30} {'N':>4}  {'neato':>6}  {'RL':>6}  {'SPC':>8}"
    print(header)
    print('-' * len(header))

    results = []
    for fname in coord_files:
        stem     = fname.replace('.coord', '')
        graphml  = os.path.join(ROME_DIR, stem + '.graphml')
        coord_f  = os.path.join(COORD_DIR, fname)

        if not os.path.exists(graphml):
            print(f"{stem:<30}  SKIP (no graphml)")
            continue

        G = nx.read_graphml(graphml)
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        rl_coords = load_coord(coord_f)
        xfn = XingLoss(G, soft=False)
        rl_x = int(xfn(rl_coords).item())

        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
            neato_coords = torch.tensor(
                [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32
            )
            neato_x = int(xfn(neato_coords).item())
            spc = (neato_x - rl_x) / max(neato_x, 1) * 100
        except Exception:
            neato_x = None
            spc = 0.0

        results.append((stem, G.number_of_nodes(), neato_x, rl_x, spc))
        neato_str = f"{neato_x:6d}" if neato_x is not None else "     ?"
        print(f"{stem:<30} {G.number_of_nodes():>4}  {neato_str}  {rl_x:6d}  {spc:+7.1f}%")

    # Summary
    valid = [(n, nx, rx, spc) for _, n, nx, rx, spc in results if nx is not None]
    if not valid:
        print("\nNo valid results.")
        return

    avg_neato = sum(nx for _, nx, _, _ in valid) / len(valid)
    avg_rl    = sum(rx for _, _, rx, _ in valid) / len(valid)
    avg_spc   = sum(spc for _, _, _, spc in valid) / len(valid)
    n_better  = sum(1 for _, nx, rx, _ in valid if rx < nx)
    n_worse   = sum(1 for _, nx, rx, _ in valid if rx > nx)
    n_equal   = sum(1 for _, nx, rx, _ in valid if rx == nx)

    print(f"\n{'='*55}")
    print(f"总图数:      {len(valid)}")
    print(f"平均 neato:  {avg_neato:.2f}")
    print(f"平均 RL:     {avg_rl:.2f}")
    print(f"平均 SPC:    {avg_spc:+.2f}%")
    print(f"优于 neato:  {n_better}/{len(valid)}")
    print(f"等于 neato:  {n_equal}/{len(valid)}")
    print(f"差于 neato:  {n_worse}/{len(valid)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
