"""
Generate layout comparison figures including GATv2-PPO column.
Output: visualizations/layout_compare_ppo_{graph}.png

Usage:
    python visualize_with_ppo.py

Put this file in your project root (same level as gnn_policy.py, xing.py, etc.)
"""
import os, sys, random, math
import torch
import torch.nn as nn
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xing import XingLoss
from gnn_policy import (
    GNNPolicy, GATv2Policy,
    build_normalized_adj, build_adj, get_node_features,
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
ROME_DIR  = os.path.join(BASE_DIR, 'rome')
VIZ_DIR   = os.path.join(BASE_DIR, 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True)

STEP_SIZE = 15.0
SIGMA     = 0.5
MAX_STEPS = 300
N_TRIALS  = 5
K_NODES   = 2

# ── Which graphs to visualize ─────────────────────────────────────────────────
SAMPLE_GRAPHS = [
    'grafo10064.39.graphml',   # neato=26, best contrast
    'grafo10084.97.graphml',   # neato=182, large graph
    'grafo10060.90.graphml',   # neato=50, medium-large
]

# ── Layout helpers ────────────────────────────────────────────────────────────

def run_graphviz(G, prog):
    pos = nx.nx_agraph.graphviz_layout(G, prog=prog)
    return torch.tensor([[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32)

def run_sa(G, n_steps=6000):
    xfn = XingLoss(G, soft=False)
    c   = run_graphviz(G, "neato")
    T, T_min, alpha = 50.0, 0.01, 0.9995
    cur = c.clone(); best_x = xfn(cur).item(); best_c = cur.clone()
    for _ in range(n_steps):
        ni = random.randint(0, G.number_of_nodes() - 1)
        nc = cur.clone()
        nc[ni, 0] += random.uniform(-10, 10)
        nc[ni, 1] += random.uniform(-10, 10)
        ox, nx_ = xfn(cur).item(), xfn(nc).item()
        if nx_ < ox or random.random() < math.exp(-(nx_ - ox) / max(T, 1e-9)):
            cur = nc
            if nx_ < best_x: best_x = nx_; best_c = nc.clone()
        if best_x == 0: break
        T = max(T * alpha, T_min)
    return best_c

def norm_c(c):
    lo, hi = c.min(), c.max()
    return (c - lo) / (hi - lo).clamp(min=1.0) * 2 - 1

# ── MLP policy ────────────────────────────────────────────────────────────────
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
def get_mlp(n):
    if n in _mlp_cache: return _mlp_cache[n]
    fname = 'policy_n40_final.pt' if n == 40 else f'policy_n{n}.pt'
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path): return None
    p = _MLP(n); p.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    p.eval(); _mlp_cache[n] = p; return p

def run_mlp(G):
    n = G.number_of_nodes(); policy = get_mlp(n)
    xfn = XingLoss(G, soft=False); init = run_graphviz(G, "neato")
    if policy is None: return init
    best_x = xfn(init).item(); best_c = init.clone()
    with torch.no_grad():
        for _ in range(N_TRIALS):
            c = init.clone(); bx = xfn(c).item()
            for _ in range(MAX_STEPS):
                nl, dm = policy(norm_c(c).flatten())
                ni = torch.distributions.Categorical(logits=nl).sample().item()
                dv = torch.distributions.Normal(dm[ni], SIGMA).sample()
                nc = c.clone(); nc[ni] = c[ni] + dv * STEP_SIZE
                nx_ = xfn(nc).item()
                if nx_ <= bx: c, bx = nc, nx_
                if bx == 0: break
            if bx < best_x: best_x = bx; best_c = c.clone()
    return best_c

# ── GNN-RL policy ─────────────────────────────────────────────────────────────
_gnn = None
def get_gnn():
    global _gnn
    if _gnn: return _gnn
    path = os.path.join(MODEL_DIR, 'gnn_policy_final.pt')
    if not os.path.exists(path): return None
    p = GNNPolicy(node_dim=3, hidden=128, n_layers=3)
    p.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    p.eval(); _gnn = p; return p

def run_gnn(G):
    policy = get_gnn(); xfn = XingLoss(G, soft=False)
    adj = build_normalized_adj(G); init = run_graphviz(G, "neato")
    if policy is None: return init
    best_x = xfn(init).item(); best_c = init.clone()
    with torch.no_grad():
        for _ in range(N_TRIALS):
            c = init.clone(); bx = xfn(c).item()
            for _ in range(MAX_STEPS):
                feat = get_node_features(G, c)
                nl, dm = policy(feat, adj)
                ni = torch.distributions.Categorical(logits=nl).sample().item()
                dv = torch.distributions.Normal(dm[ni], SIGMA).sample()
                nc = c.clone(); nc[ni] = c[ni] + dv * STEP_SIZE
                nx_ = xfn(nc).item()
                if nx_ <= bx: c, bx = nc, nx_
                if bx == 0: break
            if bx < best_x: best_x = bx; best_c = c.clone()
    return best_c

# ── GATv2-PPO policy ──────────────────────────────────────────────────────────
_gatv2 = None
def get_gatv2():
    global _gatv2
    if _gatv2: return _gatv2
    path = os.path.join(MODEL_DIR, 'gnn_ppo_final.pt')
    if not os.path.exists(path): return None
    p = GATv2Policy(node_dim=3, hidden=128, n_layers=3, n_heads=4, dropout=0.0)
    p.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    p.eval(); _gatv2 = p; return p

def run_gatv2(G):
    policy = get_gatv2(); xfn = XingLoss(G, soft=False)
    adj = build_adj(G); init = run_graphviz(G, "neato")
    if policy is None: return init
    N = G.number_of_nodes(); k = min(K_NODES, N)
    best_x = xfn(init).item(); best_c = init.clone()
    with torch.no_grad():
        for _ in range(N_TRIALS):
            c = init.clone(); bx = xfn(c).item()
            for _ in range(MAX_STEPS):
                feat = get_node_features(G, c)
                nl, dm, _ = policy(feat, adj)
                probs = torch.softmax(nl, dim=0)
                idxs  = torch.multinomial(probs, k, replacement=False)
                nc = c.clone()
                for idx in idxs:
                    dv = torch.distributions.Normal(dm[idx], SIGMA).sample()
                    nc[idx] = c[idx] + dv * STEP_SIZE
                nx_ = xfn(nc).item()
                if nx_ <= bx: c, bx = nc, nx_
                if bx == 0: break
            if bx < best_x: best_x = bx; best_c = c.clone()
    return best_c

# ── Draw one panel ────────────────────────────────────────────────────────────
def draw_panel(ax, G, coords, title, color):
    pos = {v: (coords[i, 0].item(), coords[i, 1].item()) for i, v in enumerate(G.nodes())}
    xfn = XingLoss(G, soft=False)
    xings = int(xfn(coords).item())
    nx.draw_networkx_edges(ax=ax, G=G, pos=pos,
                           edge_color='#555555', width=0.8, alpha=0.65)
    nx.draw_networkx_nodes(ax=ax, G=G, pos=pos,
                           node_size=20, node_color=color, alpha=0.9)
    ax.set_title(f"{title}\nxings={xings}", fontsize=9, fontweight='bold')
    ax.axis('off')
    return xings

# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading models...", flush=True)
gnn_ok   = get_gnn()   is not None
gatv2_ok = get_gatv2() is not None
print(f"  GNN-RL:    {'OK' if gnn_ok   else 'NOT FOUND'}", flush=True)
print(f"  GATv2-PPO: {'OK' if gatv2_ok else 'NOT FOUND'}", flush=True)

for fname in SAMPLE_GRAPHS:
    path = os.path.join(ROME_DIR, fname)
    if not os.path.exists(path):
        print(f"  skip {fname} (not found in rome/)", flush=True)
        continue

    G   = nx.read_graphml(path)
    G   = nx.convert_node_labels_to_integers(G, ordering="sorted")
    n   = G.number_of_nodes()
    xfn = XingLoss(G, soft=False)
    print(f"\n{fname}  (n={n})", flush=True)

    # Compute all 6 layouts
    layouts = [
        ('neato',      run_graphviz(G, "neato"), '#4C72B0'),
        ('sfdp',       run_graphviz(G, "sfdp"),  '#DD8452'),
        ('SA',         run_sa(G),                '#55A868'),
        ('RL-MLP',     run_mlp(G),               '#C44E52'),
        ('GNN-RL',     run_gnn(G),               '#8172B3'),
        ('GATv2-PPO',  run_gatv2(G),             '#E67E22'),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    fig.suptitle(
        f"{fname}  (n={n} nodes, {G.number_of_edges()} edges)",
        fontsize=11, fontweight='bold', y=1.02
    )
    for ax, (name, coords, color) in zip(axes, layouts):
        xings = draw_panel(ax, G, coords, name, color)
        print(f"  {name:12s}: {xings} crossings", flush=True)

    plt.tight_layout()
    stem = fname.replace('.graphml', '')
    out  = os.path.join(VIZ_DIR, f'layout_compare_ppo_{stem}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> saved: {out}", flush=True)

print("\nDone! Images in:", VIZ_DIR, flush=True)
