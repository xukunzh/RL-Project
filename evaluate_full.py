"""
Full evaluation on the Rome test set (grafo10000 – grafo10100).

Compares seven methods:
  1. neato       – Graphviz spring-electrical layout (primary baseline)
  2. sfdp        – Graphviz force-directed layout
  3. SmartGD     – GNN-based method (xing column from metrics.csv)
  4. SA          – Simulated Annealing from neato start
  5. RL-MLP      – Per-node-size MLP policy (REINFORCE, train_all_sizes.py)
  6. GNN-RL      – Size-agnostic GNN policy (REINFORCE, train_gnn.py)
  7. GATv2-PPO   – GATv2 + PPO + multi-node + curriculum (train_gnn_ppo.py)

Relative improvement vs neato (per graph):
    rel_i = (ours_i - neato_i) / max(ours_i, neato_i)
Mean over test set; negative = better than neato.

Usage:
    python evaluate_full.py
Outputs:
    eval_full.csv   (per-graph numbers)
    (console)       summary table
"""
import os
import sys
import random
import math

import torch
import torch.nn as nn
import networkx as nx
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xing import XingLoss
from gnn_policy import (
    GNNPolicy, GATv2Policy,
    build_normalized_adj, build_adj, get_node_features,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
ROME_DIR     = os.path.join(BASE_DIR, 'rome')
METRICS_CSV  = os.path.join(BASE_DIR, 'metrics.csv')
OUT_CSV      = os.path.join(BASE_DIR, 'eval_full.csv')

# ── Rollout settings ──────────────────────────────────────────────────────────
SA_STEPS      = 8000   # SA iterations per graph
RL_MLP_TRIALS = 5      # independent rollouts per test graph (MLP)
GNN_TRIALS    = 5      # independent rollouts per test graph (GNN)
RL_MAX_STEPS  = 300    # steps per RL rollout
STEP_SIZE     = 15.0
SIGMA         = 0.5


# ── SmartGD lookup (from metrics.csv) ────────────────────────────────────────

def load_smartgd(metrics_csv: str) -> dict:
    """
    Returns {graph_stem: xing_count} e.g. {"grafo10000.38": 13.0}
    The 'xing' column stores SmartGD edge crossings.
    """
    df = pd.read_csv(metrics_csv)
    # Column names: graph_id, stress, xing, ...
    xing_col = 'xing'
    id_col   = 'graph_id'
    return {
        str(row[id_col]): float(row[xing_col])
        for _, row in df.iterrows()
    }


# ── neato / sfdp ─────────────────────────────────────────────────────────────

def run_graphviz(G: nx.Graph, prog: str) -> float:
    xfn    = XingLoss(G, soft=False)
    pos    = nx.nx_agraph.graphviz_layout(G, prog=prog)
    coords = torch.tensor(
        [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32
    )
    return xfn(coords).item()


# ── Simulated annealing ───────────────────────────────────────────────────────

def run_sa(G: nx.Graph, n_steps: int = SA_STEPS) -> float:
    """SA from neato initial layout, random single-node moves."""
    xfn    = XingLoss(G, soft=False)
    pos    = nx.nx_agraph.graphviz_layout(G, prog="neato")
    coords = torch.tensor(
        [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32
    )
    T, T_min, alpha = 50.0, 0.01, 0.9995
    cur    = coords.clone()
    best_x = xfn(cur).item()

    for _ in range(n_steps):
        ni = random.randint(0, G.number_of_nodes() - 1)
        nc = cur.clone()
        nc[ni, 0] += random.uniform(-10, 10)
        nc[ni, 1] += random.uniform(-10, 10)
        old_x = xfn(cur).item()
        new_x = xfn(nc).item()
        if new_x < old_x or random.random() < math.exp(-(new_x - old_x) / max(T, 1e-9)):
            cur = nc
            best_x = min(best_x, new_x)
        if best_x == 0:
            break
        T = max(T * alpha, T_min)
    return best_x


# ── RL-MLP (per-size policy) ──────────────────────────────────────────────────

class _MLPPolicy(nn.Module):
    """Reconstruct architecture from train_all_sizes.py for weight loading."""
    def __init__(self, n_nodes: int, hidden: int = 256):
        super().__init__()
        self.n_nodes = n_nodes
        self.encoder = nn.Sequential(
            nn.Linear(n_nodes * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),      nn.ReLU(),
        )
        self.node_head  = nn.Linear(hidden, n_nodes)
        self.delta_head = nn.Linear(hidden, n_nodes * 2)

    def forward(self, x):
        h = self.encoder(x)
        return self.node_head(h), self.delta_head(h).reshape(self.n_nodes, 2)


_mlp_cache: dict = {}

def _get_mlp(n_nodes: int):
    if n_nodes in _mlp_cache:
        return _mlp_cache[n_nodes]
    fname = 'policy_n40_final.pt' if n_nodes == 40 else f'policy_n{n_nodes}.pt'
    path  = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        return None
    p = _MLPPolicy(n_nodes)
    p.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    p.eval()
    _mlp_cache[n_nodes] = p
    return p


def _normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    c_min, c_max = coords.min(), coords.max()
    return (coords - c_min) / (c_max - c_min).clamp(min=1.0) * 2 - 1


def run_rl_mlp(G: nx.Graph, n_trials: int = RL_MLP_TRIALS) -> float:
    """Run the per-size MLP policy; falls back to SA if no model exists."""
    n      = G.number_of_nodes()
    policy = _get_mlp(n)
    if policy is None:
        return run_sa(G)

    xfn         = XingLoss(G, soft=False)
    pos         = nx.nx_agraph.graphviz_layout(G, prog="neato")
    init_coords = torch.tensor(
        [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32
    )
    overall_best = xfn(init_coords).item()

    with torch.no_grad():
        for _ in range(n_trials):
            coords = init_coords.clone()
            best_x = xfn(coords).item()
            for _ in range(RL_MAX_STEPS):
                flat = _normalize_coords(coords).flatten()
                nl, dm = policy(flat)
                ni  = torch.distributions.Categorical(logits=nl).sample().item()
                dv  = torch.distributions.Normal(dm[ni], SIGMA).sample()
                nc  = coords.clone()
                nc[ni] = coords[ni] + dv * STEP_SIZE
                nx_ = xfn(nc).item()
                if nx_ <= best_x:
                    coords, best_x = nc, nx_
                if best_x == 0:
                    break
            overall_best = min(overall_best, best_x)
    return overall_best


# ── GNN-RL (size-agnostic policy) ────────────────────────────────────────────

_gnn_policy = None

def _get_gnn():
    global _gnn_policy
    if _gnn_policy is not None:
        return _gnn_policy
    path = os.path.join(MODEL_DIR, 'gnn_policy_final.pt')
    if not os.path.exists(path):
        return None
    p = GNNPolicy(node_dim=3, hidden=128, n_layers=3)
    p.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    p.eval()
    _gnn_policy = p
    return p


def run_gnn_rl(G: nx.Graph, n_trials: int = GNN_TRIALS) -> float:
    """Run the GCN-REINFORCE policy; falls back to SA if model not found."""
    policy = _get_gnn()
    if policy is None:
        print("  [GNN model not found – falling back to SA]", flush=True)
        return run_sa(G)

    adj_norm    = build_normalized_adj(G)
    xfn         = XingLoss(G, soft=False)
    pos         = nx.nx_agraph.graphviz_layout(G, prog="neato")
    init_coords = torch.tensor(
        [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32
    )
    overall_best = xfn(init_coords).item()

    with torch.no_grad():
        for _ in range(n_trials):
            coords = init_coords.clone()
            best_x = xfn(coords).item()
            for _ in range(RL_MAX_STEPS):
                features    = get_node_features(G, coords)
                nl, dm      = policy(features, adj_norm)
                ni          = torch.distributions.Categorical(logits=nl).sample().item()
                dv          = torch.distributions.Normal(dm[ni], SIGMA).sample()
                nc          = coords.clone()
                nc[ni]      = coords[ni] + dv * STEP_SIZE
                new_x       = xfn(nc).item()
                if new_x <= best_x:
                    coords, best_x = nc, new_x
                if best_x == 0:
                    break
            overall_best = min(overall_best, best_x)
    return overall_best


# ── GATv2-PPO policy ──────────────────────────────────────────────────────────

_gatv2_policy = None

def _get_gatv2():
    global _gatv2_policy
    if _gatv2_policy is not None:
        return _gatv2_policy
    path = os.path.join(MODEL_DIR, 'gnn_ppo_final.pt')
    if not os.path.exists(path):
        return None
    p = GATv2Policy(node_dim=3, hidden=128, n_layers=3, n_heads=4, dropout=0.0)
    p.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    p.eval()
    _gatv2_policy = p
    return p


def run_gatv2_ppo(G: nx.Graph, n_trials: int = GNN_TRIALS) -> float:
    """
    Run the GATv2+PPO policy (train_gnn_ppo.py).
    Each trial uses multi-node actions (k=2) for inference as well.
    Falls back to GNN-RL result if model not found.
    """
    policy = _get_gatv2()
    if policy is None:
        return run_gnn_rl(G, n_trials)   # graceful fallback

    adj         = build_adj(G)
    xfn         = XingLoss(G, soft=False)
    pos         = nx.nx_agraph.graphviz_layout(G, prog="neato")
    init_coords = torch.tensor(
        [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32
    )
    overall_best = xfn(init_coords).item()
    N            = G.number_of_nodes()
    k_nodes      = min(2, N)

    with torch.no_grad():
        for _ in range(n_trials):
            coords = init_coords.clone()
            best_x = xfn(coords).item()
            for _ in range(RL_MAX_STEPS):
                features   = get_node_features(G, coords)
                nl, dm, _  = policy(features, adj)   # ignore value at inference
                probs      = torch.softmax(nl, dim=0)
                node_ids   = torch.multinomial(probs, k_nodes, replacement=False)
                nc         = coords.clone()
                for idx in node_ids:
                    dv     = torch.distributions.Normal(dm[idx], SIGMA).sample()
                    nc[idx] = coords[idx] + dv * STEP_SIZE
                new_x = xfn(nc).item()
                if new_x <= best_x:
                    coords, best_x = nc, new_x
                if best_x == 0:
                    break
            overall_best = min(overall_best, best_x)
    return overall_best


# ── Relative improvement helper ───────────────────────────────────────────────

def rel_improvement(ours: float, baseline: float) -> float:
    """
    (ours - baseline) / max(ours, baseline)
    Negative = better than baseline.
    Returns 0.0 when both are 0 (already crossing-free).
    """
    denom = max(ours, baseline, 1.0)
    return (ours - baseline) / denom


# ── Main evaluation ───────────────────────────────────────────────────────────

def main():
    # Load SmartGD reference crossings
    smartgd_map = load_smartgd(METRICS_CSV)

    # Load test graphs
    all_files  = sorted(f for f in os.listdir(ROME_DIR) if f.endswith('.graphml'))
    test_items = []
    for fname in all_files:
        try:
            idx = int(fname.split('.')[0].replace('grafo', ''))
        except ValueError:
            continue
        if not (10000 <= idx <= 10102):
            continue
        try:
            G = nx.read_graphml(os.path.join(ROME_DIR, fname))
            G = nx.convert_node_labels_to_integers(G, ordering="sorted")
            test_items.append((fname, G))
        except Exception:
            continue

    print(f"Test graphs found: {len(test_items)}", flush=True)

    # Check which models are available
    gnn_available   = os.path.exists(os.path.join(MODEL_DIR, 'gnn_policy_final.pt'))
    gatv2_available = os.path.exists(os.path.join(MODEL_DIR, 'gnn_ppo_final.pt'))
    print(f"GNN-RL model available:    {gnn_available}", flush=True)
    print(f"GATv2-PPO model available: {gatv2_available}", flush=True)

    hdr = (f"{'Graph':<35} {'N':>4} {'neato':>6} {'sfdp':>6} {'SmGD':>6} "
           f"{'SA':>6} {'MLP':>6} {'GNN':>6} {'PPO':>6}")
    print(f"\n{hdr}")
    print("-" * 89, flush=True)

    rows = []
    for fname, G in test_items:
        n      = G.number_of_nodes()
        stem   = fname.replace('.graphml', '')

        neato_x   = run_graphviz(G, "neato")
        sfdp_x    = run_graphviz(G, "sfdp")
        sa_x      = run_sa(G)
        mlp_x     = run_rl_mlp(G)
        gnn_x     = run_gnn_rl(G)   if gnn_available   else float('nan')
        gatv2_x   = run_gatv2_ppo(G) if gatv2_available else float('nan')
        smartgd_x = smartgd_map.get(stem, float('nan'))

        row = {
            'graph':      fname,
            'n_nodes':    n,
            'neato':      neato_x,
            'sfdp':       sfdp_x,
            'smartgd':    smartgd_x,
            'sa':         sa_x,
            'rl_mlp':     mlp_x,
            'gnn_rl':     gnn_x,
            'gatv2_ppo':  gatv2_x,
        }
        rows.append(row)

        def fmt(v): return f"{v:6.0f}" if not math.isnan(v) else "   N/A"
        print(
            f"{fname:<35} {n:4d} {neato_x:6.0f} {sfdp_x:6.0f} "
            f"{fmt(smartgd_x)} {sa_x:6.0f} {mlp_x:6.0f} {fmt(gnn_x)} {fmt(gatv2_x)}",
            flush=True,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{'='*89}", flush=True)
    print(f"Saved: {OUT_CSV}\n", flush=True)

    methods = ['sfdp', 'smartgd', 'sa', 'rl_mlp', 'gnn_rl', 'gatv2_ppo']
    labels  = ['sfdp', 'SmartGD', 'SA', 'RL-MLP', 'GNN-RL', 'GATv2-PPO']

    print(f"{'Method':<10} {'Avg Xings':>10} {'vs neato (↓ better)':>22}", flush=True)
    print("-" * 46, flush=True)
    print(f"{'neato':<10} {df['neato'].mean():10.2f}", flush=True)
    for col, label in zip(methods, labels):
        valid = df.dropna(subset=[col])
        if valid.empty:
            print(f"{label:<10} {'N/A':>10}", flush=True)
            continue
        avg_x = valid[col].mean()
        # Per-graph relative improvement vs neato
        diffs = [
            rel_improvement(row[col], row['neato'])
            for _, row in valid.iterrows()
        ]
        avg_rel = sum(diffs) / len(diffs) * 100
        print(f"{label:<10} {avg_x:10.2f}  {avg_rel:+.2f}%", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
