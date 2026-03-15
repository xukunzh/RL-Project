import os, sys, random
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xing import XingLoss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROME_DIR = os.path.join(BASE_DIR, 'rome')
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
        self.node_head  = nn.Linear(hidden, n_nodes)      # which node to move
        self.delta_head = nn.Linear(hidden, n_nodes * 2)  # how much to move [dx, dy]

    def forward(self, coords_flat):
        h = self.encoder(coords_flat)
        return self.node_head(h), self.delta_head(h).reshape(self.n_nodes, 2)


def normalize_coords(coords):
    c_min = coords.min()
    c_max = coords.max()
    return (coords - c_min) / (c_max - c_min).clamp(min=1.0) * 2 - 1


def run_episode(G, policy, xing_hard, xing_soft,
                step_size=15.0, max_steps=200, sigma=0.5,
                alpha=1.0, beta=5.0, entropy_coef=0.01):
    """
    Run one episode on graph G.
    Reward = alpha * (soft_before - soft_after)   # dense signal
           + beta  * (hard_before - hard_after)   # true crossing change
    """
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    coords = torch.tensor(
        [[pos[v][0], pos[v][1]] for v in G.nodes()], dtype=torch.float32)

    initial_xing = xing_hard(coords).item()
    if initial_xing == 0:
        dummy = torch.tensor(0.0, requires_grad=True)
        return dummy * 0, 0, 0, 0

    log_probs, entropies, rewards = [], [], []
    best_xing   = initial_xing
    best_coords = coords.clone()

    for _ in range(max_steps):
        coords_flat = normalize_coords(coords).flatten().detach()
        node_logits, delta_mu = policy(coords_flat)

        node_dist = torch.distributions.Categorical(logits=node_logits)
        node_idx  = node_dist.sample()
        lp_node   = node_dist.log_prob(node_idx)
        ent_node  = node_dist.entropy()

        move_dist = torch.distributions.Normal(delta_mu[node_idx], sigma)
        delta     = move_dist.sample()
        lp_move   = move_dist.log_prob(delta).sum()
        ent_move  = move_dist.entropy().sum()

        new_coords = coords.clone()
        new_coords[node_idx] = coords[node_idx] + delta * step_size

        soft_reward = alpha * (xing_soft(coords).item() - xing_soft(new_coords).item())
        hard_reward = beta  * (xing_hard(coords).item() - xing_hard(new_coords).item())
        reward      = soft_reward + hard_reward

        coords = new_coords.detach()
        hard_after = xing_hard(coords).item()
        if hard_after < best_xing:
            best_xing   = hard_after
            best_coords = new_coords.clone()

        log_probs.append(lp_node + lp_move)
        entropies.append(ent_node + ent_move)
        rewards.append(reward)

        if best_xing == 0:
            break

    # Discounted returns
    G_t, returns = 0.0, []
    for r in reversed(rewards):
        G_t = r + 0.99 * G_t
        returns.insert(0, G_t)
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1 and returns.std() > 1e-6:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss  = -sum(lp * R for lp, R in zip(log_probs, returns))
    loss -= entropy_coef * sum(entropies)

    return loss, xing_hard(best_coords).item(), initial_xing, sum(rewards)


def load_graphs(rome_dir, n_nodes, max_graphs=80, max_idx=9999):
    """Load graphs with exactly n_nodes nodes from training set."""
    all_files = sorted([f for f in os.listdir(rome_dir) if f.endswith('.graphml')])
    result = []
    print(f"Loading {n_nodes}-node graphs...", flush=True)

    for fname in all_files:
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

    print(f"Loaded {len(result)} graphs", flush=True)
    return result


def train(n_nodes=40, n_epochs=2000, lr=3e-4):
    graphs = load_graphs(ROME_DIR, n_nodes=n_nodes, max_graphs=80)
    if not graphs:
        print("No graphs found!")
        return

    policy    = PolicyNet(n_nodes=n_nodes, hidden=256)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    history   = []

    for epoch in range(n_epochs):
        fname, G  = random.choice(graphs)
        xing_hard = XingLoss(G, soft=False)
        xing_soft = XingLoss(G, soft=True)

        loss, final_xing, initial_xing, _ = run_episode(
            G, policy, xing_hard, xing_soft,
            step_size=15.0, max_steps=200, sigma=0.5,
            alpha=1.0, beta=5.0, entropy_coef=0.01,
        )

        if initial_xing > 0:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            pct = (initial_xing - final_xing) / initial_xing * 100
            history.append(pct)

            if epoch % 50 == 0:
                recent = sum(history[-50:]) / max(len(history[-50:]), 1)
                print(f"Epoch {epoch:5d} | {fname} | "
                      f"init={initial_xing:.0f} -> final={final_xing:.0f} | "
                      f"improve={pct:+.1f}% | recent50={recent:+.1f}%",
                      flush=True)

        if epoch > 0 and epoch % 500 == 0:
            ckpt = os.path.join(MODEL_DIR, f'policy_n{n_nodes}_epoch{epoch}.pt')
            torch.save(policy.state_dict(), ckpt)
            print(f"  [checkpoint: {ckpt}]", flush=True)

    path = os.path.join(MODEL_DIR, f'policy_n{n_nodes}_final.pt')
    torch.save(policy.state_dict(), path)
    print(f"\nSaved: {path}", flush=True)


if __name__ == "__main__":
    train(n_nodes=40, n_epochs=2000, lr=3e-4)