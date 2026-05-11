"""
Generate .coord files for all 101 test graphs (grafo10000 – grafo10100).

Each .coord file contains the (x, y) coordinates of every node in the best
layout found by the GATv2-PPO policy (or GNN-RL / SA as fallback).

Output format (one node per line, space-separated):
    x1 y1
    x2 y2
    ...

Output files (in the current directory by default, or OUTPUT_DIR):
    grafo10000.38.coord
    grafo10001.32.coord
    ...

Overlap check:
    If any two nodes have Euclidean distance < OVERLAP_THRESHOLD, the nodes
    are spread out by adding small random perturbations until the check passes.
    This avoids the 50% crossing-count penalty for overlapping nodes.

Usage:
    python generate_coords.py [output_dir]
    python generate_coords.py coords_submission/
"""
import os
import sys
import random
import math

import torch
import networkx as nx

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xing import XingLoss
from gnn_policy import (
    GATv2Policy, GNNPolicy,
    build_adj, build_normalized_adj, get_node_features,
)

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
ROME_DIR          = os.path.join(BASE_DIR, 'rome')
MODEL_DIR         = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR        = sys.argv[1] if len(sys.argv) > 1 else BASE_DIR

OVERLAP_THRESHOLD = 5.0   # only catch truly collocated nodes (neato already ensures visual separation)
N_TRIALS          = 5     # initial rollout trials per graph
MAX_TRIALS        = 50    # keep retrying until better than neato (up to this many)
MAX_STEPS         = 500   # steps per trial
STEP_SIZE         = 15.0
SIGMA             = 0.5
K_NODES           = 2     # multi-node for GATv2-PPO

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_gatv2_policy():
    path = os.path.join(MODEL_DIR, 'gnn_ppo_final.pt')
    if not os.path.exists(path):
        return None
    p = GATv2Policy(node_dim=3, hidden=128, n_layers=3, n_heads=4, dropout=0.0)
    p.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    p.eval()
    return p


def load_gnn_policy():
    path = os.path.join(MODEL_DIR, 'gnn_policy_final.pt')
    if not os.path.exists(path):
        return None
    p = GNNPolicy(node_dim=3, hidden=128, n_layers=3)
    p.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    p.eval()
    return p


# ── Overlap check & fix ───────────────────────────────────────────────────────

def has_overlap(coords: torch.Tensor, threshold: float = OVERLAP_THRESHOLD) -> bool:
    """Return True if any two nodes are closer than threshold."""
    N = coords.size(0)
    for i in range(N):
        for j in range(i + 1, N):
            dist = (coords[i] - coords[j]).norm().item()
            if dist < threshold:
                return True
    return False


def fix_overlap(coords: torch.Tensor,
                threshold: float = OVERLAP_THRESHOLD,
                max_iter: int = 200) -> torch.Tensor:
    """
    Iteratively perturb overlapping nodes until all pairwise distances exceed
    threshold.  Uses small random displacements (magnitude = threshold * 2).
    """
    coords = coords.clone()
    for _ in range(max_iter):
        if not has_overlap(coords, threshold):
            break
        N = coords.size(0)
        for i in range(N):
            for j in range(i + 1, N):
                dist = (coords[i] - coords[j]).norm().item()
                if dist < threshold:
                    # Push node i away from node j
                    direction = coords[i] - coords[j]
                    norm = direction.norm().clamp(min=1e-6)
                    coords[i] = coords[i] + (direction / norm) * threshold * 1.5
    return coords


# ── Layout runners ────────────────────────────────────────────────────────────

def get_neato_coords(G: nx.Graph) -> torch.Tensor:
    """Run neato (no timeout — correctness matters more than speed here)."""
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    return torch.tensor(
        [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32
    )


def _run_one_trial(policy, adj, xfn, init_coords, k):
    """One greedy rollout; returns (best_xing, best_coords) over all steps."""
    coords            = init_coords.clone()
    cur_x             = xfn(coords).item()
    trial_best_x      = cur_x
    trial_best_coords = coords.clone()
    with torch.no_grad():
        for _ in range(MAX_STEPS):
            feat  = get_node_features(None, coords) if False else get_node_features(
                      adj._G if hasattr(adj, '_G') else None, coords)
            nl, dm, _ = policy(feat, adj)
            probs = torch.softmax(nl, dim=0)
            idxs  = torch.multinomial(probs, k, replacement=False)
            nc    = coords.clone()
            for idx in idxs:
                dv      = torch.distributions.Normal(dm[idx], SIGMA).sample()
                nc[idx] = coords[idx] + dv * STEP_SIZE
            new_x = xfn(nc).item()
            if new_x < trial_best_x:
                trial_best_x      = new_x
                trial_best_coords = nc.clone()
            if new_x <= cur_x:
                coords, cur_x = nc, new_x
            if cur_x == 0:
                break
    return trial_best_x, trial_best_coords


def run_gatv2_ppo(G: nx.Graph, policy: GATv2Policy) -> torch.Tensor:
    """
    Run rollouts until we beat neato or reach MAX_TRIALS.
    Always returns the best result seen; never worse than neato.
    """
    adj         = build_adj(G)
    xfn         = XingLoss(G, soft=False)
    init_coords = get_neato_coords(G)
    neato_xing  = xfn(init_coords).item()
    best_xing   = neato_xing
    best_coords = init_coords.clone()
    N           = G.number_of_nodes()
    k           = min(K_NODES, N)

    # Attach G to adj so _run_one_trial can call get_node_features
    # (simpler: just pass G explicitly via a closure)
    def one_trial():
        coords            = init_coords.clone()
        cur_x             = xfn(coords).item()
        trial_best_x      = cur_x
        trial_best_coords = coords.clone()
        with torch.no_grad():
            for _ in range(MAX_STEPS):
                feat  = get_node_features(G, coords)
                nl, dm, _ = policy(feat, adj)
                probs = torch.softmax(nl, dim=0)
                idxs  = torch.multinomial(probs, k, replacement=False)
                nc    = coords.clone()
                for idx in idxs:
                    dv      = torch.distributions.Normal(dm[idx], SIGMA).sample()
                    nc[idx] = coords[idx] + dv * STEP_SIZE
                new_x = xfn(nc).item()
                if new_x < trial_best_x:
                    trial_best_x      = new_x
                    trial_best_coords = nc.clone()
                if new_x <= cur_x:
                    coords, cur_x = nc, new_x
                if cur_x == 0:
                    break
        return trial_best_x, trial_best_coords

    for trial_num in range(1, MAX_TRIALS + 1):
        tx, tc = one_trial()
        if tx < best_xing:
            best_xing   = tx
            best_coords = tc.clone()
        # Stop early if we've beaten neato and done at least N_TRIALS
        if best_xing < neato_xing and trial_num >= N_TRIALS:
            break

    # Hard guarantee: never worse than neato
    if best_xing >= neato_xing:
        best_coords = init_coords.clone()
    return best_coords


def run_gnn_rl(G: nx.Graph, policy: GNNPolicy) -> torch.Tensor:
    """Best layout across N_TRIALS independent rollouts using GNN-RL."""
    adj_norm     = build_normalized_adj(G)
    xfn          = XingLoss(G, soft=False)
    init_coords  = get_neato_coords(G)
    best_xing    = xfn(init_coords).item()
    best_coords  = init_coords.clone()

    with torch.no_grad():
        for _ in range(N_TRIALS):
            coords  = init_coords.clone()
            cur_x   = xfn(coords).item()
            trial_best_x      = cur_x
            trial_best_coords = coords.clone()
            for _ in range(MAX_STEPS):
                feat = get_node_features(G, coords)
                nl, dm = policy(feat, adj_norm)
                ni  = torch.distributions.Categorical(logits=nl).sample().item()
                dv  = torch.distributions.Normal(dm[ni], SIGMA).sample()
                nc  = coords.clone()
                nc[ni] = coords[ni] + dv * STEP_SIZE
                new_x = xfn(nc).item()
                if new_x < trial_best_x:
                    trial_best_x      = new_x
                    trial_best_coords = nc.clone()
                if new_x <= cur_x:
                    coords, cur_x = nc, new_x
                if cur_x == 0:
                    break
            if trial_best_x < best_xing:
                best_xing   = trial_best_x
                best_coords = trial_best_coords.clone()

    # Guarantee we never return worse than neato
    if xfn(best_coords).item() > xfn(init_coords).item():
        best_coords = init_coords.clone()
    return best_coords


def run_sa(G: nx.Graph) -> torch.Tensor:
    """Simulated annealing fallback."""
    xfn    = XingLoss(G, soft=False)
    coords = get_neato_coords(G)
    T, T_min, alpha = 50.0, 0.01, 0.9995
    best_x    = xfn(coords).item()
    best_coords = coords.clone()

    for _ in range(8000):
        ni  = random.randint(0, G.number_of_nodes() - 1)
        nc  = coords.clone()
        nc[ni, 0] += random.uniform(-10, 10)
        nc[ni, 1] += random.uniform(-10, 10)
        old_x = xfn(coords).item()
        new_x = xfn(nc).item()
        if new_x < old_x or random.random() < math.exp(-(new_x - old_x) / max(T, 1e-9)):
            coords = nc
            if new_x < best_x:
                best_x, best_coords = new_x, nc.clone()
        if best_x == 0:
            break
        T = max(T * alpha, T_min)
    return best_coords


def _sa_refine(G: nx.Graph, init_coords: torch.Tensor) -> torch.Tensor:
    """
    Short SA pass starting from an existing layout (e.g. RL output).
    Uses a lower initial temperature so it refines rather than disrupts.
    """
    xfn    = XingLoss(G, soft=False)
    coords = init_coords.clone()
    T, T_min, alpha = 5.0, 0.01, 0.999
    best_x      = xfn(coords).item()
    best_coords = coords.clone()

    for _ in range(3000):
        ni  = random.randint(0, G.number_of_nodes() - 1)
        nc  = coords.clone()
        nc[ni, 0] += random.uniform(-8, 8)
        nc[ni, 1] += random.uniform(-8, 8)
        old_x = xfn(coords).item()
        new_x = xfn(nc).item()
        if new_x < old_x or random.random() < math.exp(-(new_x - old_x) / max(T, 1e-9)):
            coords = nc
            if new_x < best_x:
                best_x, best_coords = new_x, nc.clone()
        if best_x == 0:
            break
        T = max(T * alpha, T_min)
    return best_coords


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load models (best available)
    gatv2 = load_gatv2_policy()
    gnn   = load_gnn_policy()
    if gatv2 is not None:
        print("Using GATv2-PPO policy  (models/gnn_ppo_final.pt)", flush=True)
    elif gnn is not None:
        print("Using GNN-RL policy     (models/gnn_policy_final.pt)", flush=True)
    else:
        print("No RL model found – using Simulated Annealing fallback.", flush=True)

    # Collect test graphs
    test_files = sorted(
        f for f in os.listdir(ROME_DIR)
        if f.endswith('.graphml') and
           10000 <= int(f.split('.')[0].replace('grafo', '')) <= 10102
    )
    print(f"Found {len(test_files)} test graphs.\n", flush=True)

    for i, fname in enumerate(test_files):
        stem     = fname.replace('.graphml', '')
        out_path = os.path.join(OUTPUT_DIR, f"{stem}.coord")

        # Resume: skip graphs already processed
        if os.path.exists(out_path):
            print(f"[{i+1:3d}/{len(test_files)}] {stem:<25}  already done, skipping",
                  flush=True)
            continue

        G = nx.read_graphml(os.path.join(ROME_DIR, fname))
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        N = G.number_of_nodes()

        # Choose best available policy
        if gatv2 is not None:
            coords = run_gatv2_ppo(G, gatv2)
        elif gnn is not None:
            coords = run_gnn_rl(G, gnn)
        else:
            coords = run_sa(G)

        # SA post-processing: refine RL layout with simulated annealing
        # Only keep SA result if it strictly improves on the RL layout
        xfn_sa   = XingLoss(G, soft=False)
        rl_xing  = xfn_sa(coords).item()
        if rl_xing > 0:
            sa_coords = _sa_refine(G, coords)
            if xfn_sa(sa_coords).item() < rl_xing:
                coords = sa_coords

        # Overlap safety check — fix, then fall back to neato if fix made things worse
        if has_overlap(coords):
            xfn_tmp    = XingLoss(G, soft=False)
            neato_tmp  = get_neato_coords(G)
            before_fix = xfn_tmp(coords).item()
            fixed      = fix_overlap(coords)
            if xfn_tmp(fixed).item() <= max(before_fix, xfn_tmp(neato_tmp).item()):
                coords = fixed
            else:
                coords = neato_tmp  # fix made it worse, just use neato

        # Write .coord file  (stem / out_path defined above)
        with open(out_path, 'w') as f:
            for row in coords.tolist():
                f.write(f"{row[0]:.6f} {row[1]:.6f}\n")

        # Progress log with crossing count
        xfn     = XingLoss(G, soft=False)
        x_count = xfn(coords).item()
        neato   = get_neato_coords(G)
        x_neato = xfn(neato).item()
        # SPC: positive = better than neato, negative = worse
        spc     = (x_neato - x_count) / max(x_neato, 1.0) * 100
        print(
            f"[{i+1:3d}/{len(test_files)}] {stem:<25}  N={N:3d}  "
            f"xings={x_count:5.0f}  neato={x_neato:5.0f}  "
            f"SPC={spc:+.1f}%  → {out_path}",
            flush=True,
        )

    print(f"\nDone. {len(test_files)} .coord files written to {OUTPUT_DIR}/",
          flush=True)

    # Pack into submission tar.gz
    import tarfile
    tar_path = os.path.join(BASE_DIR, 'coords_submission.tar.gz')
    with tarfile.open(tar_path, 'w:gz') as tar:
        for fname in sorted(os.listdir(OUTPUT_DIR)):
            if fname.endswith('.coord'):
                tar.add(os.path.join(OUTPUT_DIR, fname), arcname=fname)
    print(f"Packed → {tar_path}", flush=True)


if __name__ == "__main__":
    main()
