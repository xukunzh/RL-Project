"""
Enhanced GNN training with four improvements over the REINFORCE baseline:

  1. PPO (Proximal Policy Optimisation)
     Clipped surrogate objective with a value-function critic and Generalised
     Advantage Estimation (GAE). More stable updates and better sample
     efficiency than plain REINFORCE.

  2. GATv2 architecture
     Graph Attention Network v2 (Brody et al. 2022) replaces the fixed GCN
     normalisation with dynamic, learned attention weights. Each node can
     focus on the neighbours most relevant for predicting its displacement.

  3. Multi-node actions (k = 2)
     At every step the agent moves TWO nodes jointly. Coordinated moves can
     resolve crossing configurations that single-node moves cannot escape.

  4. Curriculum learning
     Training begins on small graphs (n ≤ 30), progressively widening to
     larger ones as epochs advance. This prevents the agent from being
     overwhelmed early and encourages effective generalisation.

Usage:
    python train_gnn_ppo.py

Model saved to: models/gnn_ppo_final.pt
Checkpoints:    models/gnn_ppo_ckNNN.pt
"""
import os
import sys
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import torch
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from torch.distributions import Categorical, Normal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xing import XingLoss
from gnn_policy import GATv2Policy, build_adj, get_node_features

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory // 1024**2} MB VRAM)")
else:
    print("Using CPU")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROME_DIR  = os.path.join(BASE_DIR, 'rome')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
# GATv2Policy architecture
HIDDEN      = 128
N_LAYERS    = 3
N_HEADS     = 4
DROPOUT     = 0.1

# Training
MAX_N_NODES = 100
MAX_IDX     = 9999
MAX_GRAPHS  = 2000
N_EPOCHS    = 2000
LR          = 3e-4

# PPO
K_EPOCHS    = 4       # PPO update epochs per collected episode
CLIP_EPS    = 0.2     # PPO clip epsilon
VF_COEF     = 0.5     # value function loss coefficient
ENT_COEF    = 0.01    # entropy bonus coefficient
GAE_LAMBDA  = 0.95    # GAE lambda (bias-variance trade-off)
GAMMA       = 0.99    # discount factor

# Episode
K_NODES     = 2       # number of nodes moved per step (multi-node action)
MAX_STEPS   = 60      # steps per episode
STEP_SIZE   = 15.0    # displacement magnitude scalar
SIGMA       = 0.5     # Gaussian action std-dev

# Reward
ALPHA       = 1.0     # soft-crossing reward weight
BETA        = 5.0     # hard-crossing reward weight

# Logging / checkpoints
SAVE_EVERY  = 100
LOG_EVERY   = 50


# ── Curriculum schedule ───────────────────────────────────────────────────────

def get_curriculum_max_nodes(epoch: int, total_epochs: int) -> int:
    """
    Stage-based curriculum: start with small graphs, expand progressively.

    Stage fractions and max node counts:
        0%–15%  : n ≤ 30   (easy — most edges short, few crossings)
        15%–30% : n ≤ 40
        30%–50% : n ≤ 55
        50%–75% : n ≤ 75
        75%–100%: n ≤ 100  (full training distribution)

    Returns the maximum allowed node count for the current epoch.
    """
    stages = [
        (0.00, 0.15, 30),
        (0.15, 0.30, 40),
        (0.30, 0.50, 55),
        (0.50, 0.75, 75),
        (0.75, 1.01, 100),
    ]
    progress = epoch / max(total_epochs - 1, 1)
    for start, end, max_n in stages:
        if progress < end:
            return max_n
    return 100


# ── Data loading ──────────────────────────────────────────────────────────────

def load_training_graphs(
    rome_dir: str,
    max_n: int     = MAX_N_NODES,
    max_idx: int   = MAX_IDX,
    max_total: int = MAX_GRAPHS,
):
    """Load GraphML files: index ≤ max_idx, node count ≤ max_n."""
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


# ── GAE ───────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards: list,
    values: list,
    gamma: float = GAMMA,
    lam: float   = GAE_LAMBDA,
):
    """
    Generalised Advantage Estimation (Schulman et al. 2016).

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t     = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
    R_t     = A_t + V(s_t)   (PPO return targets for value head)

    Returns:
        advantages : torch.Tensor [T]
        returns    : torch.Tensor [T]
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    last_adv   = 0.0

    for t in reversed(range(T)):
        next_val = values[t + 1] if t < T - 1 else 0.0
        delta    = rewards[t] + gamma * next_val - values[t]
        advantages[t] = delta + gamma * lam * last_adv
        last_adv = advantages[t].item()

    returns = advantages + torch.tensor(values, dtype=torch.float32)
    return advantages, returns


# ── Episode rollout (multi-node actions) ─────────────────────────────────────

def run_episode(
    G:             nx.Graph,
    policy:        GATv2Policy,
    xh:            XingLoss,
    xs:            XingLoss,
    adj:           torch.Tensor,
    fname:         str,                       # key for neato cache
    neato_cache:   dict,                      # shared lazily-filled cache
    k_nodes:       int   = K_NODES,
    step_size:     float = STEP_SIZE,
    max_steps:     int   = MAX_STEPS,
    sigma:         float = SIGMA,
    alpha:         float = ALPHA,
    beta:          float = BETA,
):
    """
    Collect one episode using the multi-node action policy.

    Multi-node action (k = k_nodes):
      1. Forward pass → node_logits [N], delta_mu [N,2], value scalar
      2. Sample k nodes without replacement using Categorical probabilities
         (log_prob approximated as sum of independent Categorical log-probs)
      3. Sample a displacement for each of the k nodes
      4. Apply all k moves simultaneously to get s_{t+1}
      5. Reward: alpha*(soft_before - soft_after) + beta*(hard_before - hard_after)

    Returns a trajectory dict and episode statistics.
    """
    # Lazily fetch neato layout with timeout (avoids Graphviz hanging)
    if fname not in neato_cache:
        pos = None
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(nx.nx_agraph.graphviz_layout, G, prog="neato")
                pos = future.result(timeout=15)
        except (FutureTimeout, Exception):
            pass
        if pos:
            neato_cache[fname] = torch.tensor(
                [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32
            ).to(adj.device)
        else:
            print(f"  [neato timeout/fail: {fname}, using random layout]", flush=True)
            neato_cache[fname] = torch.rand(
                G.number_of_nodes(), 2, device=adj.device) * 100
    coords = neato_cache[fname].clone()

    init_xing  = xh(coords).item()
    if init_xing == 0:
        return None, 0.0, 0.0   # already crossing-free

    # Clip k to available nodes; scale max_steps with graph size so large
    # graphs get enough exploration (each node moved ~3 times on average)
    N        = G.number_of_nodes()
    k_nodes  = min(k_nodes, N)
    max_steps = max(max_steps, N * 3 // k_nodes)

    trajectory = {
        'features':     [],   # [T]  each: [N, 3]
        'node_indices': [],   # [T]  each: [k] LongTensor
        'deltas':       [],   # [T]  each: [k, 2]
        'log_probs':    [],   # [T]  float (old log-probs for PPO ratio)
        'values':       [],   # [T]  float
        'rewards':      [],   # [T]  float
    }

    best_xing  = init_xing
    best_coords = coords.clone()

    # Pre-compute crossing values for the initial state (reused as "before" each step)
    soft_before = xs(coords).item()
    hard_before = xh(coords).item()

    for _ in range(max_steps):
        features = get_node_features(G, coords).to(adj.device)  # [N, 3]
        with torch.no_grad():
            node_logits, delta_mu, value = policy(features, adj)

        # ── Node selection (k nodes, without replacement) ──
        node_probs   = torch.softmax(node_logits, dim=0)  # [N]
        node_dist    = Categorical(probs=node_probs)
        # torch.multinomial: sample k distinct indices
        node_indices = torch.multinomial(node_probs, k_nodes, replacement=False)

        # Approximate log_prob (independent Categorical; without-replacement
        # exact computation is intractable but this is a standard approximation)
        lp = sum(node_dist.log_prob(idx) for idx in node_indices)

        # ── Displacement sampling ──
        deltas   = []
        new_coords = coords.clone()
        for idx in node_indices:
            move_dist = Normal(delta_mu[idx], sigma)
            dv        = move_dist.sample()
            lp        = lp + move_dist.log_prob(dv).sum()
            deltas.append(dv)
            new_coords[idx] = coords[idx] + dv * step_size

        # ── Reward (reuse before-values, only compute after) ──
        soft_after  = xs(new_coords).item()
        hard_after  = xh(new_coords).item()
        reward      = alpha * (soft_before - soft_after) + beta * (hard_before - hard_after)

        coords = new_coords.detach()

        # Cache after-values as before-values for next step
        soft_before = soft_after
        hard_before = hard_after

        if hard_after < best_xing:
            best_xing   = hard_after
            best_coords = coords.clone()

        # ── Store transition ──
        trajectory['features'].append(features)
        trajectory['node_indices'].append(node_indices.detach())
        trajectory['deltas'].append(torch.stack(deltas).detach())   # [k, 2]
        trajectory['log_probs'].append(lp.item())
        trajectory['values'].append(value.item())
        trajectory['rewards'].append(reward)

        if best_xing == 0:
            break

    return trajectory, best_xing, init_xing


# ── PPO update ────────────────────────────────────────────────────────────────

def ppo_update(
    policy:    GATv2Policy,
    optimizer: optim.Adam,
    trajectory: dict,
    adj:        torch.Tensor,
    gamma:     float = GAMMA,
    lam:       float = GAE_LAMBDA,
    k_epochs:  int   = K_EPOCHS,
    clip_eps:  float = CLIP_EPS,
    vf_coef:   float = VF_COEF,
    ent_coef:  float = ENT_COEF,
    sigma:     float = SIGMA,
):
    """
    PPO update on the collected trajectory.

    PPO clipped surrogate objective (Schulman et al., 2017):
        r_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)
        L_CLIP = E[ min(r_t A_t,  clip(r_t, 1-eps, 1+eps) A_t) ]
        L_VF   = E[ (V(s_t) - R_t)^2 ]
        L_ENT  = E[ H(pi(·|s_t)) ]
        loss   = -L_CLIP + vf_coef * L_VF - ent_coef * L_ENT

    Performs k_epochs passes over the trajectory with gradient clipping.
    """
    # ── Compute GAE advantages and returns (on GPU) ──────────────────────────
    advantages, returns = compute_gae(
        trajectory['rewards'], trajectory['values'], gamma, lam
    )
    advantages = advantages.to(adj.device)
    returns    = returns.to(adj.device)
    # Normalise advantages
    if advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    old_log_probs = torch.tensor(trajectory['log_probs'], dtype=torch.float32,
                                  device=adj.device)
    T = len(trajectory['rewards'])

    # ── K PPO epochs ─────────────────────────────────────────────────────────
    for _ in range(k_epochs):
        new_log_probs_list = []
        new_values_list    = []
        entropies_list     = []

        for t in range(T):
            features     = trajectory['features'][t].to(adj.device)   # [N, 3]
            node_indices = trajectory['node_indices'][t].to(adj.device)
            deltas       = trajectory['deltas'][t].to(adj.device)      # [k, 2]

            node_logits, delta_mu, value = policy(features, adj)
            node_dist = Categorical(logits=node_logits)

            lp  = torch.tensor(0.0)
            ent = node_dist.entropy()

            for j, idx in enumerate(node_indices):
                lp  = lp + node_dist.log_prob(idx)
                mv  = Normal(delta_mu[idx], sigma)
                lp  = lp + mv.log_prob(deltas[j]).sum()
                ent = ent + mv.entropy().sum()

            new_log_probs_list.append(lp)
            new_values_list.append(value)
            entropies_list.append(ent)

        new_log_probs = torch.stack(new_log_probs_list)   # [T]
        new_values    = torch.stack(new_values_list)       # [T]
        entropies     = torch.stack(entropies_list)        # [T]

        # ── PPO clipped surrogate ─────────────────────────────────────────
        ratio      = torch.exp(new_log_probs - old_log_probs.detach())
        adv        = advantages.detach()
        surr1      = ratio * adv
        surr2      = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # ── Value function loss ───────────────────────────────────────────
        value_loss = F.mse_loss(new_values, returns.detach())

        # ── Entropy bonus ─────────────────────────────────────────────────
        entropy_loss = -entropies.mean()

        loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    # 1. Load all training graphs
    print("Loading training graphs...", flush=True)
    raw_graphs = load_training_graphs(ROME_DIR)
    print(f"Loaded {len(raw_graphs)} graphs (n ≤ {MAX_N_NODES}, idx ≤ {MAX_IDX})",
          flush=True)

    # 2. Pre-compute graph-level data (fixed topology); move to GPU.
    #    Neato layouts are cached lazily (on first episode use), so startup
    #    is fast and each graph only pays the Graphviz cost once per run.
    print(f"Pre-computing adjacency matrices and XingLoss objects (device={DEVICE})...",
          flush=True)
    graph_data = []
    for fname, G in raw_graphs:
        adj = build_adj(G).to(DEVICE)
        xh  = XingLoss(G, soft=False, device=DEVICE)
        xs  = XingLoss(G, soft=True,  device=DEVICE)
        graph_data.append((fname, G, xh, xs, adj))

    # Lazily populated neato layout cache: fname → init_coords tensor
    _neato_cache: dict[str, torch.Tensor] = {}
    print(f"Done. {len(graph_data)} graphs ready.\n", flush=True)

    # Group graphs by size for efficient curriculum lookup
    by_size: dict[int, list] = {}
    for entry in graph_data:
        n = entry[1].number_of_nodes()
        by_size.setdefault(n, []).append(entry)

    # 3. Initialise GATv2Policy and optimiser; move policy to GPU
    policy    = GATv2Policy(node_dim=3, hidden=HIDDEN, n_layers=N_LAYERS,
                            n_heads=N_HEADS, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # Auto-resume from latest checkpoint
    def _ckpt_epoch(fname):
        return int(fname.replace('gnn_ppo_ck', '').replace('.pt', '').replace('_exit', ''))
    ckpts = sorted(
        (f for f in os.listdir(MODEL_DIR) if f.startswith('gnn_ppo_ck')),
        key=_ckpt_epoch,
    )
    start_epoch = 0
    if ckpts:
        latest = os.path.join(MODEL_DIR, ckpts[-1])
        policy.load_state_dict(torch.load(latest, map_location='cpu',
                                          weights_only=True))
        start_epoch = _ckpt_epoch(ckpts[-1])
        print(f"Resumed from {latest} (epoch {start_epoch})", flush=True)

    # 4. Training loop
    history: list[float] = []

    print(f"Training GATv2 + PPO for {N_EPOCHS} epochs "
          f"(k_nodes={K_NODES}, K_EPOCHS={K_EPOCHS})...", flush=True)
    print(
        f"{'Epoch':>6}  {'MaxN':>4}  {'n':>4}  {'init':>6}  "
        f"{'best':>6}  {'improv':>7}  {'rec50':>7}",
        flush=True,
    )
    print("-" * 62, flush=True)

    # Heartbeat thread: prints a line every 60s so we can detect hangs
    _current_epoch = [start_epoch]
    _stop_heartbeat = threading.Event()
    def _heartbeat():
        while not _stop_heartbeat.wait(60):
            print(f"  [alive] epoch={_current_epoch[0]}  {time.strftime('%H:%M:%S')}",
                  flush=True)
    threading.Thread(target=_heartbeat, daemon=True).start()

    epoch = start_epoch
    try:
        for epoch in range(start_epoch, N_EPOCHS):
            _current_epoch[0] = epoch
            # ── Curriculum: only sample from graphs ≤ current max size ──────
            max_n    = get_curriculum_max_nodes(epoch, N_EPOCHS)
            eligible = [e for e in graph_data if e[1].number_of_nodes() <= max_n]
            if not eligible:
                eligible = graph_data

            fname, G, xh, xs, adj = random.choice(eligible)

            # ── Collect episode ──────────────────────────────────────────────
            trajectory, best_xing, init_xing = run_episode(
                G, policy, xh, xs, adj, fname, _neato_cache
            )

            if trajectory is None or not trajectory['rewards']:
                continue   # graph had no crossings or episode produced nothing

            # ── PPO update ───────────────────────────────────────────────────
            ppo_update(policy, optimizer, trajectory, adj)

            if init_xing > 0:
                pct = (init_xing - best_xing) / init_xing * 100
                history.append(pct)

            if epoch % LOG_EVERY == 0 and history:
                recent = sum(history[-50:]) / min(len(history), 50)
                n      = G.number_of_nodes()
                print(
                    f"{epoch:6d}  {max_n:4d}  {n:4d}  {init_xing:6.0f}  "
                    f"{best_xing:6.0f}  {pct:+6.1f}%  {recent:+6.1f}%",
                    flush=True,
                )

            # ── Checkpoint ───────────────────────────────────────────────────
            if epoch > 0 and epoch % SAVE_EVERY == 0:
                ckpt = os.path.join(MODEL_DIR, f'gnn_ppo_ck{epoch}.pt')
                torch.save(policy.state_dict(), ckpt)
                print(f"  [checkpoint → {ckpt}]", flush=True)

    finally:
        _stop_heartbeat.set()
        print(f"  [EXIT] stopped at epoch={epoch}  {time.strftime('%H:%M:%S')}",
              flush=True)
        # Save emergency checkpoint so progress is never fully lost
        ckpt = os.path.join(MODEL_DIR, f'gnn_ppo_ck{epoch}_exit.pt')
        torch.save(policy.state_dict(), ckpt)
        print(f"  [EXIT] emergency checkpoint → {ckpt}", flush=True)

    # Final save (only reached if loop completes normally)
    path = os.path.join(MODEL_DIR, 'gnn_ppo_final.pt')
    torch.save(policy.state_dict(), path)
    print(f"\nFinal model saved → {path}", flush=True)


if __name__ == "__main__":
    train()
