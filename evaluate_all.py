import os, sys, random, math
import torch
import torch.nn as nn
import networkx as nx
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xing import XingLoss

# Paths (relative to this file)
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
ROME_DIR  = os.path.join(BASE_DIR, 'rome')


class PolicyNet(nn.Module):
    """MLP policy: input node coords, output node selection + movement."""
    def __init__(self, n_nodes, hidden=256):
        super().__init__()
        self.n_nodes = n_nodes
        self.encoder = nn.Sequential(
            nn.Linear(n_nodes * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),      nn.ReLU(),
        )
        self.node_head  = nn.Linear(hidden, n_nodes)      # which node to move
        self.delta_head = nn.Linear(hidden, n_nodes * 2)  # how much to move

    def forward(self, x):
        h = self.encoder(x)
        return self.node_head(h), self.delta_head(h).reshape(self.n_nodes, 2)


def normalize_coords(coords):
    c_min = coords.min()
    c_max = coords.max()
    return (coords - c_min) / (c_max - c_min).clamp(min=1.0) * 2 - 1


# Cache loaded models to avoid reloading
_policy_cache = {}

def get_policy(n_nodes):
    if n_nodes in _policy_cache:
        return _policy_cache[n_nodes]
    # n=40 uses the longer-trained final model
    fname = 'policy_n40_final.pt' if n_nodes == 40 else f'policy_n{n_nodes}.pt'
    path  = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        return None
    p = PolicyNet(n_nodes, hidden=256)
    p.load_state_dict(torch.load(path, map_location='cpu'))
    p.eval()
    _policy_cache[n_nodes] = p
    return p


def run_neato(G):
    xfn    = XingLoss(G, soft=False)
    pos    = nx.nx_agraph.graphviz_layout(G, prog="neato")
    coords = torch.tensor([[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32)
    return xfn(coords).item()


def run_sfdp(G):
    xfn    = XingLoss(G, soft=False)
    pos    = nx.nx_agraph.graphviz_layout(G, prog="sfdp")
    coords = torch.tensor([[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32)
    return xfn(coords).item()


def run_sa(G, n_steps=8000):
    """Simulated annealing from neato initial layout."""
    xfn    = XingLoss(G, soft=False)
    pos    = nx.nx_agraph.graphviz_layout(G, prog="neato")
    coords = torch.tensor([[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32)
    T, T_min, alpha = 50.0, 0.01, 0.9995
    cur, best_x = coords.clone(), xfn(coords).item()
    for _ in range(n_steps):
        ni = random.randint(0, len(G.nodes()) - 1)
        nc = cur.clone()
        nc[ni, 0] += random.uniform(-10, 10)
        nc[ni, 1] += random.uniform(-10, 10)
        ox, nx_ = xfn(cur).item(), xfn(nc).item()
        if nx_ < ox or random.random() < math.exp(-(nx_ - ox) / max(T, 1e-9)):
            cur = nc
            best_x = min(best_x, nx_)
        if best_x == 0:
            break
        T = max(T * alpha, T_min)
    return best_x


def run_rl(G, n_trials=5, step_size=15.0, max_steps=300, sigma=0.5):
    """
    RL rollout: sample n_trials times, keep best result.
    Falls back to SA if no model exists for this graph size.
    """
    n      = G.number_of_nodes()
    policy = get_policy(n)
    if policy is None:
        return run_sa(G)

    xfn         = XingLoss(G, soft=False)
    pos         = nx.nx_agraph.graphviz_layout(G, prog="neato")
    init_coords = torch.tensor([[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32)
    overall_best = xfn(init_coords).item()

    with torch.no_grad():
        for _ in range(n_trials):
            coords = init_coords.clone()
            best_x = xfn(coords).item()
            for _ in range(max_steps):
                flat = normalize_coords(coords).flatten()
                nl, dm = policy(flat)
                ni  = torch.distributions.Categorical(logits=nl).sample().item()
                dv  = torch.distributions.Normal(dm[ni], sigma).sample()
                nc  = coords.clone()
                nc[ni] = coords[ni] + dv * step_size
                nx_ = xfn(nc).item()
                if nx_ <= best_x:
                    coords, best_x = nc, nx_
                if best_x == 0:
                    break
            overall_best = min(overall_best, best_x)

    return overall_best


# ── Load test graphs (grafo10000 ~ grafo10100) ──
all_files  = sorted([f for f in os.listdir(ROME_DIR) if f.endswith('.graphml')])
test_files = []
for fname in all_files:
    try:
        idx = int(fname.split('.')[0].replace('grafo', ''))
    except:
        continue
    if 10000 <= idx <= 10100:
        try:
            G = nx.read_graphml(os.path.join(ROME_DIR, fname))
            G = nx.convert_node_labels_to_integers(G, ordering="sorted")
            test_files.append((fname, G))
        except:
            continue

print(f"Total test graphs: {len(test_files)}", flush=True)
print(f"\n{'Graph':<35} {'N':>4} {'neato':>6} {'sfdp':>6} {'SA':>6} {'RL':>6}")
print("-" * 65)

results = []
for fname, G in test_files:
    n      = G.number_of_nodes()
    neato_ = run_neato(G)
    sfdp_  = run_sfdp(G)
    sa_    = run_sa(G)
    rl_    = run_rl(G)
    results.append({'graph': fname, 'n_nodes': n,
                    'neato': neato_, 'sfdp': sfdp_, 'sa': sa_, 'rl': rl_})
    print(f"{fname:<35} {n:>4} {neato_:>6.0f} {sfdp_:>6.0f} {sa_:>6.0f} {rl_:>6.0f}",
          flush=True)

df = pd.DataFrame(results)
print("\n" + "=" * 65)
for method in ['sfdp', 'sa', 'rl']:
    diffs = [(row['neato'] - row[method]) / max(row['neato'], row[method], 1)
             for _, row in df.iterrows()]
    print(f"{method.upper():>4}: avg_xing={df[method].mean():.2f}, "
          f"improvement_vs_neato={sum(diffs)/len(diffs)*100:+.2f}%")
print(f"neato: avg_xing={df['neato'].mean():.2f}")

df.to_csv(os.path.join(BASE_DIR, 'eval_all.csv'), index=False)
print("\nSaved: eval_all.csv", flush=True)