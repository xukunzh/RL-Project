"""
Train the GNN policy with REINFORCE on the Rome graph training set.

Key differences from train_all_sizes.py (per-size MLP):
  - One model for ALL graph sizes (size-agnostic via GNN)
  - Trains on every graph in the training set (up to max_n_nodes)
  - Uses graph adjacency structure: GNN identifies crossing-prone nodes
  - Adjacency matrices precomputed once (graph topology never changes)

Usage:
    python train_gnn.py
Model saved to: models/gnn_policy_final.pt
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

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROME_DIR  = os.path.join(BASE_DIR, 'rome')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_N_NODES   = 100    # include all graph sizes in training
MAX_IDX       = 9999   # training set: grafo0001 – grafo9999
MAX_GRAPHS    = 2000   # cap on number of training graphs loaded
N_EPOCHS      = 5000   # training epochs (one episode per epoch)
LR            = 3e-4
GAMMA         = 0.99   # discount factor
ENTROPY_COEF  = 0.01   # entropy bonus coefficient
ALPHA         = 1.0    # weight for soft-crossing reward
BETA          = 5.0    # weight for hard-crossing reward
STEP_SIZE     = 15.0   # magnitude scalar for node displacement
MAX_STEPS     = 120    # steps per episode
SIGMA         = 0.5    # std-dev of movement Normal distribution
SAVE_EVERY    = 1000   # checkpoint interval (epochs)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_training_graphs(
    rome_dir: str,
    max_n: int = MAX_N_NODES,
    max_idx: int = MAX_IDX,
    max_total: int = MAX_GRAPHS,
):
    """
    Load all GraphML files from rome_dir whose index ≤ max_idx and
    node count ≤ max_n.  Returns list of (filename, nx.Graph).
    """
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
            if G.number_of_nodes() <= max_n and G.number_of_edges() > 0:
                result.append((fname, G))
        except Exception:
            continue
        if len(result) >= max_total:
            break
    return result


# ── Episode ───────────────────────────────────────────────────────────────────

def run_episode(
    G: nx.Graph,
    policy: GNNPolicy,
    xh: XingLoss,         # hard (discrete) crossing loss
    xs: XingLoss,         # soft (differentiable) crossing loss
    adj_norm: torch.Tensor,
    step_size: float = STEP_SIZE,
    max_steps: int = MAX_STEPS,
    sigma: float = SIGMA,
    alpha: float = ALPHA,
    beta: float = BETA,
    entropy_coef: float = ENTROPY_COEF,
):
    """
    Run one REINFORCE episode on graph G.

    Reward at each step:
        r = alpha * (soft_before – soft_after)
          + beta  * (hard_before – hard_after)

    Returns:
        loss         – REINFORCE policy gradient loss (scalar tensor, has grad)
        best_xing    – lowest hard crossing count achieved during the episode
        init_xing    – crossing count at episode start (neato layout)
        total_reward – sum of undiscounted rewards
    """
    # Start from neato layout
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    coords = torch.tensor(
        [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32
    )

    init_xing = xh(coords).item()
    if init_xing == 0:
        dummy = torch.tensor(0.0, requires_grad=True)
        return dummy * 0, 0, 0, 0

    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    rewards:   list[float]        = []

    best_xing  = init_xing
    best_coords = coords.clone()

    for _ in range(max_steps):
        features = get_node_features(G, coords)            # [N, 3]
        node_logits, delta_mu = policy(features, adj_norm) # [N], [N, 2]

        # Sample which node to move
        node_dist = Categorical(logits=node_logits)
        node_idx  = node_dist.sample()
        lp_node   = node_dist.log_prob(node_idx)
        ent_node  = node_dist.entropy()

        # Sample displacement for the chosen node
        move_dist = Normal(delta_mu[node_idx], sigma)
        delta     = move_dist.sample()
        lp_move   = move_dist.log_prob(delta).sum()
        ent_move  = move_dist.entropy().sum()

        # Apply move
        new_coords           = coords.clone()
        new_coords[node_idx] = coords[node_idx] + delta * step_size

        # Compute reward
        soft_reward = alpha * (xs(coords).item() - xs(new_coords).item())
        hard_reward = beta  * (xh(coords).item() - xh(new_coords).item())
        reward      = soft_reward + hard_reward

        coords = new_coords.detach()

        hard_now = xh(coords).item()
        if hard_now < best_xing:
            best_xing   = hard_now
            best_coords = new_coords.clone()

        log_probs.append(lp_node + lp_move)
        entropies.append(ent_node + ent_move)
        rewards.append(reward)

        if best_xing == 0:
            break

    # Compute discounted returns
    G_t, returns = 0.0, []
    for r in reversed(rewards):
        G_t = r + GAMMA * G_t
        returns.insert(0, G_t)
    ret = torch.tensor(returns, dtype=torch.float32)

    # Normalise returns (variance reduction baseline)
    if len(ret) > 1 and ret.std() > 1e-6:
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

    # REINFORCE loss with entropy bonus
    policy_loss = -sum(lp * R for lp, R in zip(log_probs, ret))
    entropy_loss = -entropy_coef * sum(entropies)
    loss = policy_loss + entropy_loss

    return loss, xh(best_coords).item(), init_xing, float(sum(rewards))


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    # 1. Load graphs
    print("Loading training graphs...", flush=True)
    raw_graphs = load_training_graphs(ROME_DIR)
    print(f"Loaded {len(raw_graphs)} graphs (n ≤ {MAX_N_NODES}, idx ≤ {MAX_IDX})",
          flush=True)

    # 2. Pre-compute graph-level data (adj_norm + XingLoss objects)
    #    Topology never changes, so this is free to cache.
    print("Pre-computing adjacency matrices and XingLoss objects...", flush=True)
    graph_data = []
    for fname, G in raw_graphs:
        adj_norm = build_normalized_adj(G)
        xh = XingLoss(G, soft=False)
        xs = XingLoss(G, soft=True)
        graph_data.append((fname, G, xh, xs, adj_norm))
    print(f"Done. {len(graph_data)} graphs ready.\n", flush=True)

    # 3. Initialise policy and optimiser
    policy    = GNNPolicy(node_dim=3, hidden=128, n_layers=3)
    optimiser = optim.Adam(policy.parameters(), lr=LR)

    # 4. Training loop
    history: list[float] = []

    print(f"Training GNN policy for {N_EPOCHS} epochs...", flush=True)
    print(f"{'Epoch':>6}  {'n':>4}  {'init':>6}  {'best':>6}  {'improv':>7}  {'rec50':>7}",
          flush=True)
    print("-" * 55, flush=True)

    for epoch in range(N_EPOCHS):
        fname, G, xh, xs, adj_norm = random.choice(graph_data)

        loss, final_xing, init_xing, _ = run_episode(
            G, policy, xh, xs, adj_norm
        )

        if init_xing > 0:
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()

            pct = (init_xing - final_xing) / init_xing * 100
            history.append(pct)

        if epoch % 200 == 0 and history:
            recent = sum(history[-50:]) / min(len(history[-50:]), 50)
            n = G.number_of_nodes()
            print(
                f"{epoch:6d}  {n:4d}  {init_xing:6.0f}  {final_xing:6.0f}"
                f"  {pct:+6.1f}%  {recent:+6.1f}%",
                flush=True,
            )

        # Checkpoint
        if epoch > 0 and epoch % SAVE_EVERY == 0:
            ckpt = os.path.join(MODEL_DIR, f'gnn_policy_epoch{epoch}.pt')
            torch.save(policy.state_dict(), ckpt)
            print(f"  [checkpoint → {ckpt}]", flush=True)

    # Final save
    path = os.path.join(MODEL_DIR, 'gnn_policy_final.pt')
    torch.save(policy.state_dict(), path)
    print(f"\nFinal model saved → {path}", flush=True)


if __name__ == "__main__":
    train()
