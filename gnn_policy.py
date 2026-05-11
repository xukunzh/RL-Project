"""
GNN-based policy for graph layout optimization.

This module provides two policy classes:
  - GNNPolicy: original GCN-based policy (REINFORCE, backward-compatible)
  - GATv2Policy: enhanced policy using Graph Attention Network v2 (for PPO)

GATv2 replaces fixed GCN normalization with dynamic, attention-weighted
neighbor aggregation. Each head independently learns which neighbours matter
most for predicting node displacement direction.

GATv2Policy also exposes a value head (critic) required by PPO.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


# ── Adjacency utilities ───────────────────────────────────────────────────────

def build_normalized_adj(G: nx.Graph) -> torch.Tensor:
    """
    Symmetrically normalised adjacency with self-loops (for GCN / GNNPolicy):
        A_norm = D^{-1/2} (A + I) D^{-1/2}
    Returns: [N, N] float tensor
    """
    N = G.number_of_nodes()
    adj = torch.zeros(N, N)
    for u, v in G.edges():
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    adj = adj + torch.eye(N)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


def build_adj(G: nx.Graph) -> torch.Tensor:
    """
    Binary adjacency with self-loops (for GATv2Policy attention masking).
    Entries are 1.0 for connected pairs (including self-loops), 0.0 otherwise.
    Returns: [N, N] float tensor
    """
    N = G.number_of_nodes()
    adj = torch.zeros(N, N)
    for u, v in G.edges():
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    adj = adj + torch.eye(N)
    return adj.clamp(max=1.0)


def get_node_features(G: nx.Graph, coords: torch.Tensor) -> torch.Tensor:
    """
    Build per-node feature matrix [N, 3]:
        [x_norm, y_norm, degree_norm]
    Coordinates normalised to [-1, 1]; degree normalised to [0, 1].
    """
    c_min = coords.min()
    c_max = coords.max()
    coords_norm = (coords - c_min) / (c_max - c_min + 1e-6) * 2 - 1  # [N, 2]

    degrees = torch.tensor(
        [G.degree(v) for v in G.nodes()], dtype=torch.float32,
        device=coords.device,                                          # match device
    )
    max_deg = degrees.max().clamp(min=1.0)
    deg_norm = (degrees / max_deg).unsqueeze(1)                        # [N, 1]

    return torch.cat([coords_norm, deg_norm], dim=1)                   # [N, 3]


# ── Original GCN policy (kept for backward compatibility) ─────────────────────

class GCNLayer(nn.Module):
    """Single GCN layer with residual: h' = LayerNorm(ReLU(W (A_norm h)))"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.linear(adj_norm @ h)))


class GNNPolicy(nn.Module):
    """
    Size-agnostic GCN policy (REINFORCE baseline).
    forward() returns (node_logits [N], delta_mu [N, 2]).
    """
    def __init__(self, node_dim: int = 3, hidden: int = 128, n_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.gnn_layers = nn.ModuleList([
            GCNLayer(hidden, hidden) for _ in range(n_layers)
        ])
        self.node_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 2),
        )

    def forward(self, node_features: torch.Tensor, adj_norm: torch.Tensor):
        h = self.input_proj(node_features)
        for layer in self.gnn_layers:
            h = h + layer(h, adj_norm)
        return self.node_head(h).squeeze(-1), self.delta_head(h)


# ── GATv2 components ──────────────────────────────────────────────────────────

class GATv2Layer(nn.Module):
    """
    Graph Attention Network v2 layer (Brody et al., 2022).

    Unlike GCN (fixed D^{-1/2} A D^{-1/2} normalisation), GATv2 learns
    *dynamic* attention coefficients per edge:

        e_{ij} = a^T  LeakyReLU( W_L h_i  +  W_R h_j )
        α_{ij} = softmax_j( e_{ij} )
        h'_i   = ReLU( LayerNorm( Σ_j α_{ij} W_V h_j ) )

    Multi-head attention: H independent heads, each of dimension head_dim.
    Output: h' ∈ R^{H * head_dim}  (= out_dim).

    Args:
        in_dim  : input feature dimension
        out_dim : output feature dimension (must be divisible by n_heads)
        n_heads : number of attention heads
        dropout : dropout applied to attention coefficients
    """
    def __init__(self, in_dim: int, out_dim: int,
                 n_heads: int = 4, dropout: float = 0.1):
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads

        # GATv2 projections
        self.W_l    = nn.Linear(in_dim, out_dim, bias=False)  # target projection
        self.W_r    = nn.Linear(in_dim, out_dim, bias=False)  # source projection
        self.att    = nn.Linear(self.head_dim, 1, bias=False) # per-head attention
        self.W_v    = nn.Linear(in_dim, out_dim, bias=False)  # value projection

        self.norm    = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h   : [N, in_dim]  node features
            adj : [N, N]       binary adjacency (with self-loops)
        Returns:
            [N, out_dim] updated node features
        """
        N = h.size(0)
        H = self.n_heads
        D = self.head_dim

        # Per-head projections: [N, H, D]
        Wlh = self.W_l(h).view(N, H, D)
        Wrh = self.W_r(h).view(N, H, D)

        # Pairwise features for all (i=target, j=source) pairs: [N, N, H, D]
        pair = Wlh.unsqueeze(1) + Wrh.unsqueeze(0)   # broadcast over i,j
        pair = F.leaky_relu(pair, negative_slope=0.2)

        # Attention scores: [N, N, H]
        e = self.att(pair).squeeze(-1)

        # Mask out non-edges (set to -inf so softmax → 0)
        mask = adj.bool().unsqueeze(-1)               # [N, N, 1]
        e    = e.masked_fill(~mask, -1e9)

        # Normalise over source dimension j (dim=1)
        alpha = F.softmax(e, dim=1)                   # [N, N, H]
        alpha = self.dropout(alpha)

        # Value aggregation
        v   = self.W_v(h).view(N, H, D)               # [N, H, D]
        # alpha [N, N, H, 1] * v [1, N, H, D] → sum over j
        agg = (alpha.unsqueeze(-1) * v.unsqueeze(0)).sum(dim=1)  # [N, H, D]
        agg = agg.view(N, -1)                          # [N, out_dim]

        return F.relu(self.norm(agg))


# ── GATv2 Policy with PPO value head ─────────────────────────────────────────

class GATv2Policy(nn.Module):
    """
    Size-agnostic GATv2 policy for graph layout optimisation (PPO).

    Enhancements over GNNPolicy:
      * GATv2 layers replace GCN (dynamic attention vs. fixed normalisation)
      * Value head for PPO critic (mean-pooled node embeddings → scalar)
      * Multi-head attention (n_heads=4) for richer structural representation

    forward() returns:
        node_logits : [N]    unnormalised log-probs for node selection
        delta_mu    : [N, 2] predicted (dx, dy) mean per node
        value       : scalar  V(s) estimate for PPO

    Args:
        node_dim : raw feature dimension (default 3: x, y, degree)
        hidden   : latent dimension (must be divisible by n_heads)
        n_layers : number of GATv2 layers
        n_heads  : attention heads per GATv2 layer
        dropout  : attention dropout rate
    """
    def __init__(self, node_dim: int = 3, hidden: int = 128,
                 n_layers: int = 3, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
        )

        self.gnn_layers = nn.ModuleList([
            GATv2Layer(hidden, hidden, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Actor heads
        self.node_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 2),
        )

        # Critic head: mean-pool node embeddings → scalar V(s)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,  # [N, node_dim]
        adj: torch.Tensor,            # [N, N] binary adjacency
    ):
        h = self.input_proj(node_features)   # [N, hidden]

        for layer in self.gnn_layers:
            h = h + layer(h, adj)            # residual GATv2

        node_logits = self.node_head(h).squeeze(-1)   # [N]
        delta_mu    = self.delta_head(h)               # [N, 2]

        # Global mean-pool → critic value
        value = self.value_head(h.mean(dim=0)).squeeze(-1)  # scalar

        return node_logits, delta_mu, value
