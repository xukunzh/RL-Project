"""
Evaluation script for the GPU-trained GNN model (gnn_gpu_final.pt).
Adds a 'GNN-GPU' column to the comparison table.

Methods compared:
  neato / sfdp / SmartGD / SA / RL-MLP / GNN-CPU / GNN-GPU

Usage:
    python evaluate_gpu.py
Output: eval_gpu.csv
"""
import os, sys, random, math
import torch
import torch.nn as nn
import networkx as nx
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xing import XingLoss
from gnn_policy import GNNPolicy, build_normalized_adj, get_node_features

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, 'models')
ROME_DIR    = os.path.join(BASE_DIR, 'rome')
METRICS_CSV = os.path.join(BASE_DIR, 'metrics.csv')
OUT_CSV     = os.path.join(BASE_DIR, 'eval_gpu.csv')

# Inference always on CPU for fairness; GPU model is loaded to CPU
DEVICE      = torch.device('cpu')

SA_STEPS     = 8000
N_TRIALS     = 5
RL_MAX_STEPS = 300
STEP_SIZE    = 15.0
SIGMA        = 0.5


# ── SmartGD ──────────────────────────────────────────────────────────────────
def load_smartgd(csv_path):
    df = pd.read_csv(csv_path)
    return {str(row['graph_id']): float(row['xing']) for _, row in df.iterrows()}


# ── Graphviz ──────────────────────────────────────────────────────────────────
def run_graphviz(G, prog):
    xfn    = XingLoss(G, soft=False)
    pos    = nx.nx_agraph.graphviz_layout(G, prog=prog)
    coords = torch.tensor([[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32)
    return xfn(coords).item()


# ── SA ────────────────────────────────────────────────────────────────────────
def run_sa(G):
    xfn    = XingLoss(G, soft=False)
    pos    = nx.nx_agraph.graphviz_layout(G, prog="neato")
    coords = torch.tensor([[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32)
    T, T_min, alpha = 50.0, 0.01, 0.9995
    cur    = coords.clone()
    best_x = xfn(cur).item()
    for _ in range(SA_STEPS):
        ni = random.randint(0, G.number_of_nodes() - 1)
        nc = cur.clone()
        nc[ni, 0] += random.uniform(-10, 10)
        nc[ni, 1] += random.uniform(-10, 10)
        ox, nx_ = xfn(cur).item(), xfn(nc).item()
        if nx_ < ox or random.random() < math.exp(-(nx_ - ox) / max(T, 1e-9)):
            cur = nc
            best_x = min(best_x, nx_)
        if best_x == 0: break
        T = max(T * alpha, T_min)
    return best_x


# ── MLP-RL ────────────────────────────────────────────────────────────────────
class _MLP(nn.Module):
    def __init__(self, n, hidden=256):
        super().__init__()
        self.n_nodes = n
        self.encoder = nn.Sequential(
            nn.Linear(n*2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.node_head  = nn.Linear(hidden, n)
        self.delta_head = nn.Linear(hidden, n * 2)
    def forward(self, x):
        h = self.encoder(x)
        return self.node_head(h), self.delta_head(h).reshape(self.n_nodes, 2)

_mlp_cache = {}
def _get_mlp(n):
    if n in _mlp_cache: return _mlp_cache[n]
    fname = 'policy_n40_final.pt' if n == 40 else f'policy_n{n}.pt'
    path  = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path): return None
    p = _MLP(n)
    p.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    p.eval(); _mlp_cache[n] = p; return p

def _norm(c):
    lo, hi = c.min(), c.max()
    return (c - lo) / (hi - lo).clamp(min=1.0) * 2 - 1

def run_mlp(G):
    n = G.number_of_nodes(); policy = _get_mlp(n)
    xfn  = XingLoss(G, soft=False)
    init = torch.tensor([[nx.nx_agraph.graphviz_layout(G, prog="neato")[v][0],
                          nx.nx_agraph.graphviz_layout(G, prog="neato")[v][1]]
                         for v in G.nodes()], dtype=torch.float32)
    if policy is None: return xfn(init).item()
    best_x = xfn(init).item()
    with torch.no_grad():
        for _ in range(N_TRIALS):
            c = init.clone(); bx = xfn(c).item()
            for _ in range(RL_MAX_STEPS):
                nl, dm = policy(_norm(c).flatten())
                ni = torch.distributions.Categorical(logits=nl).sample().item()
                dv = torch.distributions.Normal(dm[ni], SIGMA).sample()
                nc = c.clone(); nc[ni] = c[ni] + dv * STEP_SIZE
                nx_ = xfn(nc).item()
                if nx_ <= bx: c, bx = nc, nx_
                if bx == 0: break
            best_x = min(best_x, bx)
    return best_x


# ── GNN rollout helper ────────────────────────────────────────────────────────
def _run_gnn_rollout(G, policy, hidden):
    """Generic rollout for any GNN policy (CPU inference)."""
    adj  = build_normalized_adj(G)
    xfn  = XingLoss(G, soft=False)
    pos  = nx.nx_agraph.graphviz_layout(G, prog="neato")
    init = torch.tensor([[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32)
    best_x = xfn(init).item()
    with torch.no_grad():
        for _ in range(N_TRIALS):
            c = init.clone(); bx = xfn(c).item()
            for _ in range(RL_MAX_STEPS):
                feat = get_node_features(G, c.cpu() if c.is_cuda else c)
                nl, dm = policy(feat, adj)
                ni = torch.distributions.Categorical(logits=nl).sample().item()
                dv = torch.distributions.Normal(dm[ni], SIGMA).sample()
                nc = c.clone(); nc[ni] = c[ni] + dv * STEP_SIZE
                nx_ = xfn(nc).item()
                if nx_ <= bx: c, bx = nc, nx_
                if bx == 0: break
            best_x = min(best_x, bx)
    return best_x


# ── GNN-CPU (128-dim, trained on CPU) ────────────────────────────────────────
_gnn_cpu = None
def run_gnn_cpu(G):
    global _gnn_cpu
    if _gnn_cpu is None:
        path = os.path.join(MODEL_DIR, 'gnn_policy_final.pt')
        if not os.path.exists(path):
            return float('nan')
        p = GNNPolicy(node_dim=3, hidden=128, n_layers=3)
        p.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        p.eval(); _gnn_cpu = p
    return _run_gnn_rollout(G, _gnn_cpu, hidden=128)


# ── GNN-GPU (256-dim, trained with GPU) ──────────────────────────────────────
_gnn_gpu = None
def run_gnn_gpu(G):
    global _gnn_gpu
    if _gnn_gpu is None:
        path = os.path.join(MODEL_DIR, 'gnn_gpu_final.pt')
        if not os.path.exists(path):
            # try latest checkpoint
            ckpts = sorted([f for f in os.listdir(MODEL_DIR)
                            if f.startswith('gnn_gpu_ck') and f.endswith('.pt')])
            if not ckpts:
                return float('nan')
            path = os.path.join(MODEL_DIR, ckpts[-1])
            print(f"  [using GPU checkpoint: {ckpts[-1]}]", flush=True)
        p = GNNPolicy(node_dim=3, hidden=128, n_layers=3)
        p.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        p.eval(); _gnn_gpu = p
    return _run_gnn_rollout(G, _gnn_gpu, hidden=256)


# ── Relative improvement ──────────────────────────────────────────────────────
def rel_imp(ours, neato):
    denom = max(ours, neato, 1.0)
    return (ours - neato) / denom


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    smartgd = load_smartgd(METRICS_CSV)

    all_files  = sorted(f for f in os.listdir(ROME_DIR) if f.endswith('.graphml'))
    test_items = []
    for fname in all_files:
        try:
            idx = int(fname.split('.')[0].replace('grafo', ''))
        except ValueError:
            continue
        if not (10000 <= idx <= 10100):
            continue
        try:
            G = nx.read_graphml(os.path.join(ROME_DIR, fname))
            G = nx.convert_node_labels_to_integers(G, ordering="sorted")
            test_items.append((fname, G))
        except Exception:
            continue

    gpu_ckpts  = sorted([f for f in os.listdir(MODEL_DIR)
                         if f.startswith('gnn_gpu_ck') and f.endswith('.pt')])
    cpu_ready  = os.path.exists(os.path.join(MODEL_DIR, 'gnn_policy_final.pt'))
    gpu_ready  = bool(gpu_ckpts) or os.path.exists(os.path.join(MODEL_DIR, 'gnn_gpu_final.pt'))

    print(f"Test graphs: {len(test_items)}", flush=True)
    print(f"GNN-CPU model: {'ready' if cpu_ready else 'NOT FOUND'}", flush=True)
    print(f"GNN-GPU model: {'ready (' + gpu_ckpts[-1] + ')' if gpu_ckpts else 'NOT FOUND (train first)'}", flush=True)
    print()

    hdr = f"{'Graph':<35} {'N':>4} {'neato':>6} {'sfdp':>6} {'SmGD':>6} {'SA':>6} {'MLP':>6} {'CPU':>6} {'GPU':>6}"
    print(hdr)
    print("-" * 90, flush=True)

    rows = []
    for fname, G in test_items:
        n       = G.number_of_nodes()
        stem    = fname.replace('.graphml', '')
        neato_x = run_graphviz(G, "neato")
        sfdp_x  = run_graphviz(G, "sfdp")
        sa_x    = run_sa(G)
        mlp_x   = run_mlp(G)
        cpu_x   = run_gnn_cpu(G) if cpu_ready else float('nan')
        gpu_x   = run_gnn_gpu(G) if gpu_ready else float('nan')
        sg_x    = smartgd.get(stem, float('nan'))

        rows.append({'graph': fname, 'n_nodes': n, 'neato': neato_x,
                     'sfdp': sfdp_x, 'smartgd': sg_x, 'sa': sa_x,
                     'rl_mlp': mlp_x, 'gnn_cpu': cpu_x, 'gnn_gpu': gpu_x})

        def fmt(v): return f"{v:6.0f}" if not math.isnan(v) else "   N/A"
        print(f"{fname:<35} {n:4d} {neato_x:6.0f} {sfdp_x:6.0f} "
              f"{fmt(sg_x)} {sa_x:6.0f} {mlp_x:6.0f} {fmt(cpu_x)} {fmt(gpu_x)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n{'='*90}", flush=True)
    print(f"{'Method':<10} {'Avg Xings':>10} {'vs neato':>22}", flush=True)
    print("-" * 45, flush=True)
    print(f"{'neato':<10} {df['neato'].mean():10.2f}", flush=True)
    for col, label in [('sfdp','sfdp'), ('smartgd','SmartGD'), ('sa','SA'),
                       ('rl_mlp','RL-MLP'), ('gnn_cpu','GNN-CPU'), ('gnn_gpu','GNN-GPU')]:
        valid = df.dropna(subset=[col])
        if valid.empty: print(f"{label:<10} {'N/A':>10}"); continue
        avg_x = valid[col].mean()
        diffs = [rel_imp(r[col], r['neato']) for _, r in valid.iterrows()]
        print(f"{label:<10} {avg_x:10.2f}  {sum(diffs)/len(diffs)*100:+.2f}%", flush=True)

    print(f"\nSaved: {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
