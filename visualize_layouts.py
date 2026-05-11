"""
Visualize graph layouts produced by each method for a few test graphs.
Generates: visualizations/layout_compare_{graph}.png
"""
import os, sys, random, math
import torch
import torch.nn as nn
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xing import XingLoss
from gnn_policy import GNNPolicy, build_normalized_adj, get_node_features

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
ROME_DIR  = os.path.join(BASE_DIR, 'rome')
VIZ_DIR   = os.path.join(BASE_DIR, 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True)

STEP_SIZE = 15.0
SIGMA     = 0.5
MAX_STEPS = 300
N_TRIALS  = 5

# ── Helpers ───────────────────────────────────────────────────────────────────

def coords_to_pos(G, coords):
    return {v: (coords[i, 0].item(), coords[i, 1].item()) for i, v in enumerate(G.nodes())}

def run_graphviz_coords(G, prog):
    pos = nx.nx_agraph.graphviz_layout(G, prog=prog)
    coords = torch.tensor([[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32)
    return coords

def run_sa_coords(G, n_steps=6000):
    xfn    = XingLoss(G, soft=False)
    coords = run_graphviz_coords(G, "neato")
    T, T_min, alpha = 50.0, 0.01, 0.9995
    cur    = coords.clone()
    best_x = xfn(cur).item()
    best_c = cur.clone()
    for _ in range(n_steps):
        ni = random.randint(0, G.number_of_nodes() - 1)
        nc = cur.clone()
        nc[ni, 0] += random.uniform(-10, 10)
        nc[ni, 1] += random.uniform(-10, 10)
        ox, nx_ = xfn(cur).item(), xfn(nc).item()
        if nx_ < ox or random.random() < math.exp(-(nx_ - ox) / max(T, 1e-9)):
            cur = nc
            if nx_ < best_x:
                best_x = nx_
                best_c = nc.clone()
        if best_x == 0: break
        T = max(T * alpha, T_min)
    return best_c

# ── MLP policy ────────────────────────────────────────────────────────────────
class _MLPPolicy(nn.Module):
    def __init__(self, n_nodes, hidden=256):
        super().__init__()
        self.n_nodes = n_nodes
        self.encoder = nn.Sequential(
            nn.Linear(n_nodes*2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.node_head  = nn.Linear(hidden, n_nodes)
        self.delta_head = nn.Linear(hidden, n_nodes * 2)
    def forward(self, x):
        h = self.encoder(x)
        return self.node_head(h), self.delta_head(h).reshape(self.n_nodes, 2)

_mlp_cache = {}
def get_mlp(n):
    if n in _mlp_cache: return _mlp_cache[n]
    fname = 'policy_n40_final.pt' if n == 40 else f'policy_n{n}.pt'
    path  = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path): return None
    p = _MLPPolicy(n)
    p.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    p.eval(); _mlp_cache[n] = p; return p

def norm_c(c):
    lo, hi = c.min(), c.max()
    return (c - lo) / (hi - lo).clamp(min=1.0) * 2 - 1

def run_mlp_coords(G):
    n = G.number_of_nodes()
    policy = get_mlp(n)
    xfn    = XingLoss(G, soft=False)
    init   = run_graphviz_coords(G, "neato")
    if policy is None: return init, xfn(init).item()
    best_x = xfn(init).item(); best_c = init.clone()
    with torch.no_grad():
        for _ in range(N_TRIALS):
            c = init.clone()
            bx = xfn(c).item()
            for _ in range(MAX_STEPS):
                nl, dm = policy(norm_c(c).flatten())
                ni  = torch.distributions.Categorical(logits=nl).sample().item()
                dv  = torch.distributions.Normal(dm[ni], SIGMA).sample()
                nc  = c.clone(); nc[ni] = c[ni] + dv * STEP_SIZE
                nx_ = xfn(nc).item()
                if nx_ <= bx: c, bx = nc, nx_
                if bx == 0: break
            if bx < best_x: best_x, best_c = bx, c.clone()
    return best_c, best_x

# ── GNN policy ────────────────────────────────────────────────────────────────
_gnn = None
def get_gnn():
    global _gnn
    if _gnn: return _gnn
    path = os.path.join(MODEL_DIR, 'gnn_policy_final.pt')
    if not os.path.exists(path): return None
    p = GNNPolicy(node_dim=3, hidden=128, n_layers=3)
    p.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    p.eval(); _gnn = p; return p

def run_gnn_coords(G):
    policy = get_gnn()
    xfn    = XingLoss(G, soft=False)
    adj    = build_normalized_adj(G)
    init   = run_graphviz_coords(G, "neato")
    if policy is None: return init, xfn(init).item()
    best_x = xfn(init).item(); best_c = init.clone()
    with torch.no_grad():
        for _ in range(N_TRIALS):
            c = init.clone(); bx = xfn(c).item()
            for _ in range(MAX_STEPS):
                feat = get_node_features(G, c)
                nl, dm = policy(feat, adj)
                ni  = torch.distributions.Categorical(logits=nl).sample().item()
                dv  = torch.distributions.Normal(dm[ni], SIGMA).sample()
                nc  = c.clone(); nc[ni] = c[ni] + dv * STEP_SIZE
                nx_ = xfn(nc).item()
                if nx_ <= bx: c, bx = nc, nx_
                if bx == 0: break
            if bx < best_x: best_x, best_c = bx, c.clone()
    return best_c, best_x

# ── Draw one panel ────────────────────────────────────────────────────────────
def draw_panel(ax, G, coords, title, crossings, color):
    pos = coords_to_pos(G, coords)
    nx.draw_networkx_edges(ax=ax, G=G, pos=pos,
                           edge_color='#555555', width=0.8, alpha=0.7)
    nx.draw_networkx_nodes(ax=ax, G=G, pos=pos,
                           node_size=20, node_color=color, alpha=0.9)
    ax.set_title(f"{title}\nxings={int(crossings)}", fontsize=9, fontweight='bold')
    ax.axis('off')

# ── Main ─────────────────────────────────────────────────────────────────────
SAMPLE_GRAPHS = [
    'grafo10003.40.graphml',   # medium, 17 crossings neato
    'grafo10057.40.graphml',   # hard, 39 crossings neato
    'grafo10064.39.graphml',   # hard, 26 crossings neato
    'grafo10060.90.graphml',   # large, 50 crossings neato
    'grafo10084.97.graphml',   # large, 182 crossings neato
]

METHODS = [
    ('neato',   '#4C72B0'),
    ('sfdp',    '#DD8452'),
    ('SA',      '#55A868'),
    ('RL-MLP',  '#C44E52'),
    ('GNN-RL',  '#8172B3'),
]

print("Generating layout visualizations...", flush=True)
gnn_available = os.path.exists(os.path.join(MODEL_DIR, 'gnn_policy_final.pt'))
print(f"GNN model: {'available' if gnn_available else 'NOT found'}", flush=True)

for fname in SAMPLE_GRAPHS:
    path = os.path.join(ROME_DIR, fname)
    if not os.path.exists(path):
        print(f"  skip {fname} (not found)", flush=True)
        continue

    G = nx.read_graphml(path)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    xfn = XingLoss(G, soft=False)
    n   = G.number_of_nodes()
    print(f"\n{fname}  (n={n})", flush=True)

    # Compute all layouts
    neato_c  = run_graphviz_coords(G, "neato")
    sfdp_c   = run_graphviz_coords(G, "sfdp")
    sa_c     = run_sa_coords(G)
    mlp_c, _ = run_mlp_coords(G)
    gnn_c, _ = run_gnn_coords(G) if gnn_available else (neato_c, xfn(neato_c).item())

    layouts = [
        ('neato',  neato_c,  '#4C72B0'),
        ('sfdp',   sfdp_c,   '#DD8452'),
        ('SA',     sa_c,     '#55A868'),
        ('RL-MLP', mlp_c,    '#C44E52'),
        ('GNN-RL', gnn_c,    '#8172B3'),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f"{fname}  (n={n} nodes, {G.number_of_edges()} edges)",
                 fontsize=11, fontweight='bold', y=1.02)

    for ax, (name, coords, color) in zip(axes, layouts):
        xings = xfn(coords).item()
        draw_panel(ax, G, coords, name, xings, color)
        print(f"  {name:8s}: {int(xings)} crossings", flush=True)

    plt.tight_layout()
    stem = fname.replace('.graphml', '')
    out  = os.path.join(VIZ_DIR, f'layout_compare_{stem}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}", flush=True)

print("\nAll done! Visualizations in:", VIZ_DIR, flush=True)
