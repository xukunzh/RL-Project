"""
Faster GNN training: focuses on graphs with n ≤ 50 nodes for quick results.
These cover the majority of the test set (n=30-42).
Large-graph generalisation is handled by RL-MLP fallback in evaluate_full.py.

Usage: python train_gnn_fast.py
Model saved to: models/gnn_policy_final.pt   (same name as full training)
Checkpoints: models/gnn_policy_ck{epoch}.pt every 500 epochs
"""
import os
import sys
import random

import torch
import torch.optim as optim
import networkx as nx
from torch.distributions import Categorical, Normal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xing import XingLoss
from gnn_policy import GNNPolicy, build_normalized_adj, get_node_features

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROME_DIR  = os.path.join(BASE_DIR, 'rome')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────
MAX_N_NODES  = 50      # focus on smaller graphs
MAX_IDX      = 9999
MAX_GRAPHS   = 800     # enough diversity without blowing memory
N_EPOCHS     = 3000
LR           = 3e-4
GAMMA        = 0.99
ENTROPY_COEF = 0.01
ALPHA        = 1.0
BETA         = 5.0
STEP_SIZE    = 15.0
MAX_STEPS    = 100
SIGMA        = 0.5
SAVE_EVERY   = 200   # save every 200 epochs so we lose less progress if killed


def load_graphs(rome_dir, max_n, max_idx, max_total):
    result = []
    for fname in sorted(os.listdir(rome_dir)):
        if not fname.endswith('.graphml'):
            continue
        try:
            idx = int(fname.split('.')[0].replace('grafo', ''))
        except ValueError:
            continue
        if idx > max_idx:
            continue
        try:
            G = nx.read_graphml(os.path.join(rome_dir, fname))
            G = nx.convert_node_labels_to_integers(G, ordering="sorted")
            n = G.number_of_nodes()
            if n <= max_n and n >= 10 and G.number_of_edges() > 0:
                result.append((fname, G))
        except Exception:
            continue
        if len(result) >= max_total:
            break
    return result


def run_episode(G, policy, xh, xs, adj_norm):
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    coords = torch.tensor(
        [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32
    )

    init_x = xh(coords).item()
    if init_x == 0:
        return torch.tensor(0.0, requires_grad=True) * 0, 0, 0

    log_probs, entropies, rewards = [], [], []
    best_x = init_x
    best_coords = coords.clone()

    for _ in range(MAX_STEPS):
        features = get_node_features(G, coords)
        nl, dm   = policy(features, adj_norm)

        nd  = Categorical(logits=nl)
        ni  = nd.sample()
        md  = Normal(dm[ni], SIGMA)
        dv  = md.sample()

        nc = coords.clone()
        nc[ni] = coords[ni] + dv * STEP_SIZE

        soft_r = ALPHA * (xs(coords).item() - xs(nc).item())
        hard_r = BETA  * (xh(coords).item() - xh(nc).item())
        reward = soft_r + hard_r

        coords = nc.detach()
        hx = xh(coords).item()
        if hx < best_x:
            best_x = hx
            best_coords = nc.clone()

        log_probs.append(nd.log_prob(ni) + md.log_prob(dv).sum())
        entropies.append(nd.entropy() + md.entropy().sum())
        rewards.append(reward)
        if best_x == 0:
            break

    G_t, returns = 0.0, []
    for r in reversed(rewards):
        G_t = r + GAMMA * G_t
        returns.insert(0, G_t)
    ret = torch.tensor(returns, dtype=torch.float32)
    if len(ret) > 1 and ret.std() > 1e-6:
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

    loss  = -sum(lp * R for lp, R in zip(log_probs, ret))
    loss -= ENTROPY_COEF * sum(entropies)
    return loss, xh(best_coords).item(), init_x


def train():
    print("Loading graphs...", flush=True)
    raw = load_graphs(ROME_DIR, MAX_N_NODES, MAX_IDX, MAX_GRAPHS)
    print(f"Loaded {len(raw)} graphs (n ≤ {MAX_N_NODES})", flush=True)

    print("Pre-computing adj + XingLoss...", flush=True)
    data = []
    for fname, G in raw:
        data.append((fname, G, XingLoss(G, soft=False), XingLoss(G, soft=True),
                     build_normalized_adj(G)))
    print(f"{len(data)} graphs ready.\n", flush=True)

    policy    = GNNPolicy(node_dim=3, hidden=128, n_layers=3)
    optimiser = optim.Adam(policy.parameters(), lr=LR)
    history: list[float] = []
    start_epoch = 0

    # Resume from latest checkpoint if available
    ckpts = sorted(
        [f for f in os.listdir(MODEL_DIR) if f.startswith('gnn_policy_ck') and f.endswith('.pt')],
        key=lambda f: int(f.replace('gnn_policy_ck','').replace('.pt',''))
    )
    if ckpts:
        latest = ckpts[-1]
        path = os.path.join(MODEL_DIR, latest)
        policy.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        # parse epoch number from filename e.g. gnn_policy_ck500.pt
        try:
            start_epoch = int(latest.replace('gnn_policy_ck','').replace('.pt',''))
        except:
            start_epoch = 0
        print(f"Resumed from {latest} (epoch {start_epoch})", flush=True)

    print(f"Training for {N_EPOCHS} epochs (start={start_epoch})...", flush=True)
    for epoch in range(start_epoch, N_EPOCHS):
        fname, G, xh, xs, adj = random.choice(data)

        loss, final_x, init_x = run_episode(G, policy, xh, xs, adj)

        if init_x > 0:
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()
            pct = (init_x - final_x) / init_x * 100
            history.append(pct)

        if epoch % 200 == 0 and history:
            recent = sum(history[-50:]) / min(len(history[-50:]), 50)
            n = G.number_of_nodes()
            print(
                f"Epoch {epoch:5d} | n={n:3d} | {init_x:.0f}→{final_x:.0f} "
                f"| {pct:+.1f}% | rec50={recent:+.1f}%",
                flush=True,
            )

        if epoch > 0 and epoch % SAVE_EVERY == 0:
            ck = os.path.join(MODEL_DIR, f'gnn_policy_ck{epoch}.pt')
            torch.save(policy.state_dict(), ck)
            print(f"  [checkpoint → {ck}]", flush=True)

    path = os.path.join(MODEL_DIR, 'gnn_policy_final.pt')
    torch.save(policy.state_dict(), path)
    print(f"\nSaved: {path}", flush=True)


if __name__ == "__main__":
    train()
