"""
GPU-accelerated GNN training for graph layout optimization.

Key differences from train_gnn_fast.py (CPU):
  - All tensors (adj_norm, coords, XingLoss edges) are moved to CUDA
  - Larger hidden dim (256) and more graphs (1500) thanks to GPU memory
  - Includes graphs up to n=100 nodes (previously limited to 50)
  - Larger batch of training steps per episode (150 steps)
  - Checkpoints every 200 epochs; auto-resumes from latest checkpoint

Usage:
    python train_gnn_gpu.py
Output: models/gnn_gpu_final.pt
        models/gnn_gpu_ck{epoch}.pt  (every 200 epochs)
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

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}", flush=True)
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROME_DIR  = os.path.join(BASE_DIR, 'rome')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_N_NODES  = 50     # same as CPU version
MAX_IDX      = 9999
MAX_GRAPHS   = 800    # same as CPU version
N_EPOCHS     = 3000   # same as CPU version
LR           = 3e-4
GAMMA        = 0.99
ENTROPY_COEF = 0.01
ALPHA        = 1.0
BETA         = 5.0
STEP_SIZE    = 15.0
MAX_STEPS    = 100    # same as CPU version
SIGMA        = 0.5
HIDDEN       = 128    # same as CPU version
N_LAYERS     = 3
SAVE_EVERY   = 200


# ── Data loading ──────────────────────────────────────────────────────────────

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
            if n >= 10 and n <= max_n and G.number_of_edges() > 0:
                result.append((fname, G))
        except Exception:
            continue
        if len(result) >= max_total:
            break
    return result


# ── Pre-compute graph data (on GPU) ───────────────────────────────────────────

def precompute(raw_graphs, device):
    """Move adj_norm and XingLoss edge tensors to GPU."""
    data = []
    for fname, G in raw_graphs:
        adj_norm = build_normalized_adj(G).to(device)          # [N, N] on GPU
        xh = XingLoss(G, soft=False, device=device)            # edges on GPU
        xs = XingLoss(G, soft=True,  device=device)            # edges on GPU
        data.append((fname, G, xh, xs, adj_norm))
    return data


# ── Episode ───────────────────────────────────────────────────────────────────

def run_episode(G, policy, xh, xs, adj_norm, device):
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    coords = torch.tensor(
        [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32
    ).to(device)                                               # coords on GPU

    init_x = xh(coords).item()
    if init_x == 0:
        return torch.tensor(0.0, requires_grad=True, device=device) * 0, 0, 0

    log_probs, entropies, rewards = [], [], []
    best_x      = init_x
    best_coords = coords.clone()

    for _ in range(MAX_STEPS):
        # get_node_features builds degree tensor on CPU, so pass coords on CPU
        features = get_node_features(G, coords.cpu()).to(device)
        nl, dm   = policy(features, adj_norm)               # GNN on GPU

        nd  = Categorical(logits=nl)
        ni  = nd.sample()
        md  = Normal(dm[ni], SIGMA)
        dv  = md.sample()

        nc       = coords.clone()
        nc[ni]   = coords[ni] + dv * STEP_SIZE

        soft_r  = ALPHA * (xs(coords).item() - xs(nc).item())
        hard_r  = BETA  * (xh(coords).item() - xh(nc).item())
        reward  = soft_r + hard_r

        coords = nc.detach()
        hx = xh(coords).item()
        if hx < best_x:
            best_x, best_coords = hx, nc.clone()

        log_probs.append(nd.log_prob(ni) + md.log_prob(dv).sum())
        entropies.append(nd.entropy() + md.entropy().sum())
        rewards.append(reward)
        if best_x == 0:
            break

    G_t, returns = 0.0, []
    for r in reversed(rewards):
        G_t = r + GAMMA * G_t
        returns.insert(0, G_t)
    ret = torch.tensor(returns, dtype=torch.float32, device=device)
    if len(ret) > 1 and ret.std() > 1e-6:
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

    loss  = -sum(lp * R for lp, R in zip(log_probs, ret))
    loss -= ENTROPY_COEF * sum(entropies)
    return loss, xh(best_coords).item(), init_x


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    # 1. Load graphs
    print("Loading graphs...", flush=True)
    raw = load_graphs(ROME_DIR, MAX_N_NODES, MAX_IDX, MAX_GRAPHS)
    print(f"Loaded {len(raw)} graphs (n ≤ {MAX_N_NODES})", flush=True)

    # 2. Pre-compute on GPU
    print("Pre-computing adj + XingLoss on GPU...", flush=True)
    data = precompute(raw, DEVICE)
    print(f"{len(data)} graphs ready on {DEVICE}.\n", flush=True)

    # 3. Build policy on GPU
    policy    = GNNPolicy(node_dim=3, hidden=HIDDEN, n_layers=N_LAYERS).to(DEVICE)
    optimiser = optim.Adam(policy.parameters(), lr=LR)
    start_epoch = 0

    # 4. Auto-resume from latest GPU checkpoint
    gpu_ckpts = sorted([f for f in os.listdir(MODEL_DIR)
                        if f.startswith('gnn_gpu_ck') and f.endswith('.pt')])
    if gpu_ckpts:
        latest = gpu_ckpts[-1]
        path   = os.path.join(MODEL_DIR, latest)
        policy.load_state_dict(
            torch.load(path, map_location=DEVICE, weights_only=True)
        )
        try:
            start_epoch = int(latest.replace('gnn_gpu_ck', '').replace('.pt', ''))
        except Exception:
            start_epoch = 0
        print(f"Resumed from {latest} (epoch {start_epoch})", flush=True)

    # 5. Training
    history: list[float] = []
    print(f"Training for {N_EPOCHS} epochs (start={start_epoch})...", flush=True)

    for epoch in range(start_epoch, N_EPOCHS):
        fname, G, xh, xs, adj_norm = random.choice(data)

        loss, final_x, init_x = run_episode(G, policy, xh, xs, adj_norm, DEVICE)

        if init_x > 0:
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()
            pct = (init_x - final_x) / init_x * 100
            history.append(pct)

        if epoch % 200 == 0 and history:
            recent = sum(history[-50:]) / min(len(history[-50:]), 50)
            n      = G.number_of_nodes()
            vram   = torch.cuda.memory_allocated(0) / 1e6 if DEVICE.type == 'cuda' else 0
            print(
                f"Epoch {epoch:5d} | n={n:3d} | {init_x:.0f}→{final_x:.0f}"
                f" | {pct:+.1f}% | rec50={recent:+.1f}%"
                + (f" | VRAM={vram:.0f}MB" if DEVICE.type == 'cuda' else ""),
                flush=True,
            )

        if epoch > 0 and epoch % SAVE_EVERY == 0:
            ck = os.path.join(MODEL_DIR, f'gnn_gpu_ck{epoch}.pt')
            torch.save(policy.state_dict(), ck)
            print(f"  [checkpoint → {ck}]", flush=True)

    path = os.path.join(MODEL_DIR, 'gnn_gpu_final.pt')
    torch.save(policy.state_dict(), path)
    print(f"\nFinal model saved → {path}", flush=True)


if __name__ == "__main__":
    train()
