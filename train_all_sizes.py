"""
Train one policy model per node size found in the test set.
Test set node sizes: 31,32,34,35,37,38,39,40,41,42,90,93,94,95,96,97,98,99
Each model trains for 1000 epochs (~10-15 min per size).
Models are saved to models/policy_n{N}.pt
"""
import os, sys, random
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xing import XingLoss

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROME_DIR  = os.path.join(BASE_DIR, 'rome')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


class PolicyNet(nn.Module):
    """MLP policy: input node coords, output node selection + movement."""
    def __init__(self, n_nodes, hidden=256):
        super().__init__()
        self.n_nodes = n_nodes
        self.encoder = nn.Sequential(
            nn.Linear(n_nodes * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),      nn.ReLU(),
        )
        self.node_head  = nn.Linear(hidden, n_nodes)
        self.delta_head = nn.Linear(hidden, n_nodes * 2)

    def forward(self, coords_flat):
        h = self.encoder(coords_flat)
        return self.node_head(h), self.delta_head(h).reshape(self.n_nodes, 2)


def normalize_coords(coords):
    c_min = coords.min()
    c_max = coords.max()
    return (coords - c_min) / (c_max - c_min).clamp(min=1.0) * 2 - 1


def run_episode(G, policy, xh, xs, step_size=15.0, max_steps=200,
                sigma=0.5, alpha=1.0, beta=5.0, entropy_coef=0.01):
    pos    = nx.nx_agraph.graphviz_layout(G, prog="neato")
    coords = torch.tensor(
        [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32)

    init_x = xh(coords).item()
    if init_x == 0:
        return torch.tensor(0.0, requires_grad=True) * 0, 0, 0, 0

    log_probs, entropies, rewards = [], [], []
    best_x, best_coords = init_x, coords.clone()

    for _ in range(max_steps):
        flat   = normalize_coords(coords).flatten().detach()
        nl, dm = policy(flat)

        nd = torch.distributions.Categorical(logits=nl)
        ni = nd.sample()
        md = torch.distributions.Normal(dm[ni], sigma)
        dv = md.sample()

        nc    = coords.clone()
        nc[ni] = coords[ni] + dv * step_size

        r = alpha * (xs(coords).item() - xs(nc).item()) + \
            beta  * (xh(coords).item() - xh(nc).item())

        coords = nc.detach()
        hx     = xh(coords).item()
        if hx < best_x:
            best_x, best_coords = hx, nc.clone()

        log_probs.append(nd.log_prob(ni) + md.log_prob(dv).sum())
        entropies.append(nd.entropy() + md.entropy().sum())
        rewards.append(r)
        if best_x == 0:
            break

    G_t, returns = 0.0, []
    for r in reversed(rewards):
        G_t = r + 0.99 * G_t
        returns.insert(0, G_t)
    ret = torch.tensor(returns, dtype=torch.float32)
    if len(ret) > 1 and ret.std() > 1e-6:
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

    loss  = -sum(lp * R for lp, R in zip(log_probs, ret))
    loss -= entropy_coef * sum(entropies)
    return loss, xh(best_coords).item(), init_x, sum(rewards)


def load_graphs(rome_dir, n_nodes, max_graphs=60, max_idx=9999):
    result = []
    for fname in sorted(os.listdir(rome_dir)):
        if not fname.endswith('.graphml'):
            continue
        try:
            idx = int(fname.split('.')[0].replace('grafo', ''))
        except:
            continue
        if idx > max_idx:
            continue
        try:
            G = nx.read_graphml(os.path.join(rome_dir, fname))
            G = nx.convert_node_labels_to_integers(G, ordering="sorted")
            if G.number_of_nodes() == n_nodes:
                result.append((fname, G))
        except:
            continue
        if len(result) >= max_graphs:
            break
    return result


def train_one_size(n_nodes, n_epochs=1000, lr=3e-4):
    graphs = load_graphs(ROME_DIR, n_nodes)
    if not graphs:
        print(f"[n={n_nodes}] No graphs found, skip", flush=True)
        return

    print(f"\n[n={n_nodes}] {len(graphs)} graphs, {n_epochs} epochs...", flush=True)
    policy    = PolicyNet(n_nodes, hidden=256)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    history   = []

    for epoch in range(n_epochs):
        fname, G = random.choice(graphs)
        xh = XingLoss(G, soft=False)
        xs = XingLoss(G, soft=True)

        loss, final_x, init_x, _ = run_episode(G, policy, xh, xs)

        if init_x > 0:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            history.append((init_x - final_x) / init_x * 100)

        if epoch % 200 == 0 and history:
            recent = sum(history[-50:]) / min(len(history[-50:]), 50)
            print(f"  epoch {epoch:5d} | {fname} | "
                  f"init={init_x:.0f} -> final={final_x:.0f} | "
                  f"recent50={recent:+.1f}%", flush=True)

    path = os.path.join(MODEL_DIR, f'policy_n{n_nodes}.pt')
    torch.save(policy.state_dict(), path)
    print(f"  Saved: {path}", flush=True)


if __name__ == "__main__":
    # All node sizes appearing in test set (grafo10000-10100)
    TEST_SIZES = [31, 32, 34, 35, 37, 38, 39, 40, 41, 42,
                  90, 93, 94, 95, 96, 97, 98, 99]

    # n=40 already trained with more epochs, skip
    SKIP = {40}

    for n in TEST_SIZES:
        if n in SKIP:
            print(f"[n={n}] already trained, skip", flush=True)
            continue
        train_one_size(n_nodes=n, n_epochs=1000, lr=3e-4)

    print("\nAll done!", flush=True)