# Graph Layout Optimization via Reinforcement Learning: Minimizing Edge Crossings

**Course:** CSA 5180 — Reinforcement Learning  
**Date:** April 2026  
**Dataset:** Rome Graph Collection  

---

## Abstract

We formulate graph layout optimization as a Reinforcement Learning (RL) problem, where the goal is to learn a policy that iteratively repositions nodes in a 2-D drawing to minimize edge crossings — a central aesthetic criterion in graph visualization. Starting from a force-directed initialization (neato), our RL agent applies node-displacement moves guided by a learned policy. We develop and compare three RL policy architectures — a per-size MLP (RL-MLP), a size-agnostic Graph Convolutional Network (GNN-RL) trained with REINFORCE, and a GATv2 policy trained with PPO (GATv2-PPO) — against four baselines. We additionally introduce four enhancements into GATv2-PPO: (1) **Proximal Policy Optimization (PPO)** for more stable training; (2) **Graph Attention Network v2 (GATv2)** replacing fixed GCN normalization with dynamic attention-weighted aggregation; (3) **multi-node actions** that move two nodes jointly per step; and (4) **curriculum learning** that begins training on small graphs and progressively scales to larger ones. On 101 test graphs from the Rome collection, our GATv2-PPO achieves a mean relative improvement of **−29.79%** vs. neato under the official evaluation metric (submitted coordinates; quick-evaluation score: −23.86%), compared to GNN-RL (−23.29%), RL-MLP (−15.49%), SA (−12.04%), and SmartGD (−3.12%).

---

## 1. Introduction

Graph visualization is a fundamental tool for making sense of relational data — from social networks and dependency graphs to circuit layouts and biological pathways. A primary measure of layout quality is the number of **edge crossings**: every crossing obscures structure and impedes readability.

Formally, given an undirected graph $G = (V, E)$, a graph **layout** assigns 2-D coordinates $x_v \in \mathbb{R}^2$ to each node $v \in V$. Classical methods such as neato and sfdp minimize a differentiable proxy objective, the graph-theoretic *stress*:
$$\text{Stress}(X) = \sum_{i < j} w_{ij}\bigl(\|x_i - x_j\| - d_{ij}\bigr)^2$$
where $d_{ij}$ is the shortest-path distance and $w_{ij} = d_{ij}^{-2}$.

However, the edge crossing count $C(X) = \#\{(e_1,e_2)\in E^2 : e_1 \text{ and } e_2 \text{ cross in layout } X\}$ is **non-differentiable** and **combinatorially defined** — it cannot be directly minimized by gradient descent. This motivates a Reinforcement Learning (RL) approach, where a policy learns to make local layout improvements through trial-and-error interaction.

This paper makes the following contributions:

1. **MDP formulation** of graph layout as a sequential decision process over node positions.
2. **Three RL policy architectures**: per-size MLP, size-agnostic GCN (REINFORCE), and GATv2 (PPO).
3. **Four enhancements**: PPO for training stability, GATv2 for richer structural representation, multi-node actions for coordinated improvement, and curriculum learning for better generalization.
4. **Comprehensive evaluation** on 101 Rome test graphs against four baselines using the official evaluation metric.

---

## 2. Problem Formulation

We model graph layout optimization as a finite-horizon Markov Decision Process (MDP):

### 2.1 State

The state $s_t$ is the current node coordinate matrix $X_t \in \mathbb{R}^{|V| \times 2}$, augmented with graph-structural features (degree, adjacency) for the GNN-based policies.

**Initial state:** The neato Graphviz layout serves as the starting point. This provides a competitive initialization: neato already minimizes stress, ensuring nodes are neither too crowded nor too spread out, and the RL agent then refines this toward fewer crossings.

### 2.2 Action Space

**Single-node action** (RL-MLP, GNN-RL): At each step the agent selects
1. A node index $i \sim \text{Categorical}(\pi_\theta(\cdot \mid s_t))$
2. A displacement $\Delta x_i \sim \mathcal{N}(\mu_\theta(i, s_t),\ \sigma^2 I_2)$

and updates $x_i^{t+1} = x_i^t + \Delta x_i \cdot \text{step\_size}$.

**Multi-node action** (GATv2-PPO): At each step the agent selects $k = 2$ nodes without replacement using multinomial sampling, then samples a displacement for each. Both moves are applied simultaneously:
$$x_{i_j}^{t+1} = x_{i_j}^t + \Delta x_{i_j} \cdot \text{step\_size}, \quad j = 1, 2$$

Multi-node actions allow the agent to resolve crossing configurations that require coordinated movement — for example, when two edges cross and both endpoints need to be repositioned to eliminate the crossing.

### 2.3 Reward

The reward combines a dense differentiable signal with the true crossing count:
$$r_t = \alpha \cdot \bigl(C_\text{soft}(s_t) - C_\text{soft}(s_{t+1})\bigr) + \beta \cdot \bigl(C(s_t) - C(s_{t+1})\bigr)$$
with $\alpha = 1.0$ and $\beta = 5.0$.

The soft crossing loss uses sigmoid-smoothed edge intersection indicators:
$$C_\text{soft}(X) = \sum_{\substack{(e_1,e_2) \in E^2 \\ \text{no shared endpoint}}} \sigma\!\left(\kappa(t_{12} - 0.5)\right) \cdot \sigma\!\left(\kappa(u_{12} - 0.5)\right)$$
where $t_{12}, u_{12}$ are parametric intersection coefficients and $\kappa = 10$ is a sharpness parameter. This provides a gradient signal even when the discrete crossing count is unchanged, which is critical for stable policy gradient training.

### 2.4 Episode Structure

Each episode runs for at most $T = 300$ steps (adaptive: $\max(300,\ 3N / k)$), terminating early if zero crossings are achieved. The **best layout observed during the episode** — not the final layout — is reported as the result. This is important because RL policies may temporarily worsen a layout during exploration.

---

## 3. Methods

### 3.1 Crossing Detection (XingLoss)

For two line segments $e_1 = (p, p+r)$ and $e_2 = (q, q+s)$ with no shared endpoint, we compute the parametric intersection parameters:
$$t = \frac{(q - p) \times s}{r \times s}, \qquad u = \frac{(q - p) \times r}{r \times s}$$
where $\times$ denotes the 2-D cross product. The segments cross if and only if $t \in (0, 1)$ and $u \in (0, 1)$.

The soft variant replaces the indicator $\mathbb{1}[t \in (0,1)]$ with $\sigma(\kappa \cdot t) \cdot \sigma(\kappa \cdot (1 - t))$, a smooth approximation that peaks at $t = 0.5$ and approaches zero at the endpoints.

All non-adjacent edge pairs are precomputed once at graph load time, making each crossing evaluation $O(|E_\text{valid}|)$ at inference (where $|E_\text{valid}|$ is the number of non-adjacent pairs, prefiltered in `__init__`).

### 3.2 Architecture 1: Per-Size MLP (RL-MLP)

A 2-layer MLP takes the flattened normalized node coordinates as input:

$$\text{Input: } [x_1^{\text{norm}}, y_1^{\text{norm}}, \ldots, x_N^{\text{norm}}, y_N^{\text{norm}}] \in \mathbb{R}^{2N}$$

Two output heads share the encoder:
- **Node head:** Linear$(256) \to \mathbb{R}^N$ (selection logits)
- **Delta head:** Linear$(256) \to \mathbb{R}^{N \times 2}$ (displacement means)

One model is trained per graph size $N$. This is the simplest RL baseline.

**Limitation:** Cannot generalize across sizes; requires a separate model for every unseen $N$.

### 3.3 Architecture 2: GCN Policy (GNN-RL)

To overcome the per-size limitation, we design a **size-agnostic** Graph Convolutional Network (Kipf & Welling, 2017) policy.

**Node features** (per node): $f_v = [x_v^{\text{norm}},\ y_v^{\text{norm}},\ \deg(v)^{\text{norm}}] \in \mathbb{R}^3$

**GCN layer** (with residual connection):
$$h_v^{(k+1)} = h_v^{(k)} + \text{ReLU}\!\left(\text{LayerNorm}\!\left(W \cdot \hat{A} h_v^{(k)}\right)\right)$$
where $\hat{A} = D^{-1/2}(A + I)D^{-1/2}$ is the symmetrically normalized adjacency with self-loops.

**Architecture:** input projection (3→128, ReLU) → 3 GCN layers (hidden=128, residual) → node head [128→64→1] + delta head [128→64→2].

The same model handles **any** graph size — the GCN operates on node features and the adjacency matrix, both of which scale with $N$ without changing any weight dimensions.

### 3.4 Architecture 3: GATv2 Policy with PPO Value Head (GATv2-PPO)

We replace the fixed GCN normalization with **Graph Attention Network v2** (GATv2; Brody et al., 2022), which computes dynamic, content-dependent attention weights.

**GATv2 layer** (multi-head, $H = 4$ heads, head dimension $D = 32$):

$$e_{ij}^{(h)} = \mathbf{a}^{(h)\top} \text{LeakyReLU}\!\left(W_L^{(h)} h_i + W_R^{(h)} h_j\right)$$
$$\alpha_{ij}^{(h)} = \frac{\exp(e_{ij}^{(h)})}{\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp(e_{ik}^{(h)})}$$
$$h_i' = \text{ReLU}\!\left(\text{LayerNorm}\!\left(\bigg\|_{h=1}^H \sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}^{(h)} W_V^{(h)} h_j\bigg\|_{h=1}^H\right)\right)$$

where $\|$ denotes multi-head concatenation and $\mathcal{N}(i)$ is the neighborhood of node $i$.

**Key advantage over GCN:** The GCN aggregates with fixed weights $\hat{A}_{ij} = 1/\sqrt{d_i d_j}$ — entirely determined by degree. GATv2 learns *which* neighbors matter for crossing prediction from the current node features, adapting attention based on the layout state at each step.

**Value head (critic):** A graph-level mean pooling of node embeddings feeds a small MLP:
$$V(s) = \text{MLP}\!\left(\frac{1}{|V|} \sum_{v \in V} h_v^{(L)}\right) \in \mathbb{R}$$
This value estimate is required by the PPO critic.

**Full architecture:** 3 → 128 (input projection) → 3 × GATv2(128, H=4) → node head + delta head + value head.

### 3.5 Training: REINFORCE (Baseline)

Both RL-MLP and GNN-RL use the REINFORCE policy gradient algorithm:

**Discounted returns:** $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$, $\gamma = 0.99$

**Normalised returns** (variance reduction): $\hat{G}_t = \dfrac{G_t - \bar{G}}{\text{std}(G) + \epsilon}$

**Loss:**
$$\mathcal{L}(\theta) = -\sum_t \log \pi_\theta(a_t \mid s_t) \cdot \hat{G}_t - \lambda_H \sum_t H(\pi_\theta(\cdot \mid s_t))$$
with entropy coefficient $\lambda_H = 0.01$. Gradient clipping: $\|\nabla\theta\|_2 \leq 1.0$.

### 3.6 Training: PPO with GAE (GATv2-PPO)

Proximal Policy Optimization (PPO; Schulman et al., 2017) improves upon REINFORCE in two key ways:
- **Clipped surrogate:** prevents destructively large policy updates
- **Value function baseline:** reduces gradient variance via the critic

**Generalized Advantage Estimation (GAE):**
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$\hat{A}_t = \sum_{\ell=0}^{T-t} (\gamma \lambda)^\ell \delta_{t+\ell}, \qquad \lambda = 0.95$$

**PPO clipped surrogate objective:**
$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}, \qquad
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t \hat{A}_t,\ \text{clip}(r_t, 1{-}\varepsilon, 1{+}\varepsilon)\hat{A}_t\right)\right]$$

**Full PPO loss:**
$$\mathcal{L}(\theta) = -L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\theta) - c_2 H(\pi_\theta)$$
with $\varepsilon = 0.2$, $c_1 = 0.5$, $c_2 = 0.01$. Each collected episode is reused for $K = 4$ gradient updates. Gradient clipping: $\|\nabla\theta\|_2 \leq 0.5$.

### 3.7 Curriculum Learning

Training begins on small, simple graphs and progressively introduces larger ones:

| Training progress | Max graph size |
|:-----------------:|:--------------:|
| 0% – 15%          | $N \leq 30$    |
| 15% – 30%         | $N \leq 40$    |
| 30% – 50%         | $N \leq 55$    |
| 50% – 75%         | $N \leq 75$    |
| 75% – 100%        | $N \leq 100$   |

**Motivation:** Small graphs have fewer crossings and shorter episodes. Early in training, when the policy is nearly random, large graphs produce overwhelming noise in the reward signal. Starting with manageable cases lets the policy learn basic displacement heuristics before encountering denser, harder graphs, improving both convergence speed and final performance.

---

## 4. Baselines

| Method | Description |
|--------|-------------|
| **neato** | Graphviz stress-minimization layout (primary baseline) |
| **sfdp** | Graphviz scalable force-directed placement |
| **SmartGD** | State-of-the-art GNN-based layout model; crossings from the GraphDrawingBenchmark |
| **SA** | Simulated Annealing from neato start; 8,000 single-node moves, $T_0 = 50$, $\alpha = 0.9995$ |

---

## 5. Experiments

### 5.1 Dataset

The **Rome graph collection** contains 11,534 undirected graphs of varying sizes and densities, derived from real-world data (mostly $N \leq 100$ nodes).

- **Training set:** grafo0001 – grafo9999 (up to 2,000 graphs used; GATv2-PPO trained on $N \leq 100$; per-size MLPs use graphs of matching size)
- **Test set:** grafo10000 – grafo10102 (101 graphs, $N \in [30, 99]$, never seen during training; the official range grafo10000–grafo10100 yields 99 graphs since grafo10073 and grafo10094 do not exist in the Rome collection — we supplement with grafo10101 and grafo10102 to reach 101 total)

The test set spans a range of sizes and densities, with edge counts ranging from $|E| = 29$ to $|E| = 160$.

### 5.2 Evaluation Metric

Following the project specification, we use **per-graph relative improvement vs. neato**, averaged over the test set:
$$\text{Improvement} = \frac{1}{|\mathcal{T}|} \sum_{g \in \mathcal{T}} \frac{C_g^{\text{ours}} - C_g^{\text{neato}}}{\max(C_g^{\text{ours}},\ C_g^{\text{neato}})}$$
Negative values indicate fewer crossings than neato (better). A value of $-0.2979$ means the method is 29.79% better than neato on average.

### 5.3 Main Results

| Method | Avg. Crossings | Mean Rel. Improvement |
|--------|:--------------:|:---------------------:|
| neato | 30.51 | 0.00% (baseline) |
| sfdp | 29.75 | −5.36% |
| SmartGD | 30.00 | −3.12% |
| SA | 28.54 | −12.04% |
| RL-MLP | 26.59 | −15.49% |
| GNN-RL | 25.43 | −23.29% |
| GATv2-PPO *(quick eval)* | 24.94 | −23.86% |
| **GATv2-PPO** *(submitted)* | **23.80** | **−29.79%** |

*Quick eval uses N\_TRIALS=5, MAX\_STEPS=300 for fair comparison with other baselines. Submitted coordinates use MAX\_TRIALS=50, MAX\_STEPS=500 with SA post-processing refinement (test-time search budget increase; no retraining).*

GATv2-PPO achieves the best result across all methods. The progression from MLP → GNN-RL → GATv2-PPO demonstrates that each architectural and algorithmic improvement contributes: structural inductive bias (GCN) provides the largest single gain, and the combination of PPO, GATv2, multi-node actions, and curriculum learning brings a further improvement over GNN-RL. With increased test-time search (50 trials, 500 steps, SA refinement), the submitted coordinates achieve **−29.79%**, a 6 percentage point gain over the quick-eval score with no additional training.

Notably, SmartGD — a specialized deep learning system pre-trained on the GraphDrawingBenchmark — is outperformed by all three of our RL methods. Our methods optimize the crossing objective directly; SmartGD uses a different training signal and is not specifically designed to minimize crossings.

### 5.4 Per-Graph Analysis

Representative test cases illustrating the range of improvement (GATv2-PPO column shows submitted coordinates):

| Graph | $N$ | neato | sfdp | SmartGD | SA | RL-MLP | GNN-RL | GATv2-PPO |
|-------|:---:|:-----:|:----:|:-------:|:--:|:------:|:------:|:---------:|
| grafo10008 | 42 | 30 | 28 | 29 | 25 | 25 | 24 | **19** |
| grafo10064 | 39 | 26 | 27 | 27 | 23 | 21 | 19 | **17** |
| grafo10028 | 96 | 104 | 111 | 106 | 95 | 93 | 89 | **80** |
| grafo10084 | 97 | 182 | 184 | 191 | 181 | 168 | 164 | **154** |
| grafo10099 | 94 | 149 | 179 | 165 | 141 | 137 | 133 | **124** |
| grafo10061 | 99 | 103 | 93 | 83 | 99 | 95 | 91 | **84** |
| grafo10086 | 34 | 1 | 1 | 1 | 0 | 0 | 0 | **0** |

Several observations:
- On small-to-medium graphs ($N \leq 42$), GATv2-PPO consistently outperforms all methods, including classical baselines and GNN-RL.
- On large dense graphs ($N \geq 90$), GATv2-PPO achieves the largest absolute crossing reductions: −28 on grafo10084, −25 on grafo10099, −19 on grafo10028. The combination of attention-based representation, PPO training, and extended test-time search generalizes well to harder instances.
- SmartGD underperforms on large dense graphs, likely because its training objectives differ from direct crossing minimization.
- grafo10086 ($N = 34$): GATv2-PPO achieves **0 crossings** (neato = 1), finding a planar embedding.

### 5.5 Visualization

Side-by-side Neato vs. GATv2-PPO layouts for all 101 test graphs are provided in the `viz_output/` directory of the repository, with crossing edges highlighted in red and crossing points marked with orange stars. Two representative examples illustrate the qualitative improvement: for grafo10064 ($N=39$), neato produces a tangled central region with 26 crossings, while GATv2-PPO spreads the graph into a cleaner structure with 17 crossings (−35%). For grafo10084 ($N=97$), the densest graph in the test set, GATv2-PPO reduces crossings from 182 to 154, the largest absolute reduction of 28 crossings across all test cases. These visualizations confirm that the RL policy learns to identify and untangle crossing-prone subregions through coordinated node displacements, rather than making random small perturbations.

---

## 6. Discussion

### Why GNN-RL Outperforms RL-MLP

The key insight is **structural inductive bias**. The GCN aggregates information from each node's neighbors through message passing, enabling the policy to reason about crossing-prone neighborhoods. A node surrounded by many crossing edges will receive strong signals from its neighbors, making it more likely to be selected for movement. The MLP, operating on flat coordinates, must learn this structural reasoning from scratch for every graph size.

Additionally, GNN-RL trains on 2,000 diverse graphs simultaneously, while each per-size MLP uses only 60–80 training graphs of its specific size. The larger and more diverse training set further helps the GNN generalize.

### Why GATv2-PPO Outperforms GNN-RL

The improvement from GNN-RL to GATv2-PPO comes from four compounding factors:

**PPO vs REINFORCE:** REINFORCE uses the full episode return as advantage, which has high variance due to cumulative reward discounting. PPO with GAE reduces this variance via the learned value function baseline. For our task, where episode rewards are small and noisy (crossing reductions of 0–3 per step), high-variance gradients cause instability. PPO's clipping additionally prevents the policy from being updated too aggressively on any single episode.

**GATv2 vs GCN:** In a GCN, the contribution of neighbor $j$ to node $i$ is fixed at $1/\sqrt{d_i d_j}$, depending only on degree. In GATv2, the attention weight $\alpha_{ij}$ depends on the current features of both $i$ and $j$, allowing the model to focus on neighbors that are currently most relevant for predicting the displacement direction.

**Multi-node actions:** Single-node moves can become trapped when two edges cross and both pairs of endpoints need to be moved simultaneously to resolve the crossing. By selecting 2 nodes jointly, the agent can make coordinated improvements unreachable by greedy single-node moves.

**Curriculum learning:** Starting with small graphs ensures that early training provides clear learning signal and establishes basic skills, leading to better convergence and generalization to larger graphs.

### Effect of Reward Shaping

Without the soft crossing loss ($\alpha = 0$), training is extremely noisy: the agent moves nodes and receives reward 0 for all steps where the discrete crossing count doesn't change, which is the majority of steps in dense graphs. The differentiable soft signal provides dense learning signal, dramatically improving convergence.

### Limitations

Our crossing evaluation runs in $O(|E|^2)$ time: for dense graphs ($|E| = 150$), each step requires $\sim\!11{,}000$ segment intersection checks. Spatial indexing (sweep line, k-d tree) could reduce this to $O(|E| \log |E|)$, enabling faster training on larger graphs. Additionally, our multi-node sampling approximates the log-probability as the sum of independent Categorical terms, which is incorrect under without-replacement sampling. The Plackett-Luce distribution provides an exact treatment and could improve the policy gradient estimates. Finally, the gap between the quick-evaluation score (−23.86%, N\_TRIALS=5) and the submitted score (−29.79%, N\_TRIALS=50 + SA refinement) highlights that the current policy has not yet converged to its full potential — richer node features (e.g., per-node current crossing count) and more training epochs could reduce reliance on extended test-time search.

---

## 7. Conclusion

We demonstrated that Reinforcement Learning can effectively reduce edge crossings in graph layouts. Starting from a simple per-size MLP baseline (−15.49% vs. neato), we progressed through a size-agnostic GCN policy with REINFORCE (−23.29%) to a GATv2 policy with PPO, multi-node actions, and curriculum learning. Under a fixed quick-evaluation budget (N\_TRIALS=5), GATv2-PPO achieves −23.86%; with increased test-time search (N\_TRIALS=50, MAX\_STEPS=500, SA post-processing), the submitted coordinates achieve **−29.79%**, the best result across all methods evaluated on 101 Rome test graphs.

The core lessons are twofold. First, **graph structure is a powerful inductive bias**: a policy that can read the adjacency information through message passing can identify crossing-prone nodes and apply appropriate displacements, transferring this skill across graph sizes without retraining. Second, **test-time search amplifies learned policies**: the same trained model, given more rollout budget at inference, delivers substantially better results — a 6 percentage point gain over the quick-evaluation score with zero additional training. The four enhancements (PPO, GATv2, multi-node, curriculum) each address a concrete limitation and together push performance beyond what GCN+REINFORCE alone achieves.

**Reproducibility:** All code is available at **https://github.com/YifanCai0309/RL-Project**. To train the GATv2-PPO model: `python train_gnn_ppo.py`. To evaluate all methods: `python evaluate_full.py`. To generate submission coordinates: `python generate_coords.py coords_submission/`. The final submitted node coordinates for all 101 test graphs are packaged in `coords_submission.tar.gz` at the repository root.

---

## References

1. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8, 229–256.
2. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR 2017*.
3. Brody, S., Alon, U., & Yahav, E. (2022). How attentive are graph attention networks? *ICLR 2022*.
4. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
5. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR 2016*.
6. Di Battista, G., Eades, P., Tamassia, R., & Tollis, I. G. (1999). *Graph Drawing: Algorithms for the Visualization of Graphs*. Prentice Hall.
7. Laman, Y., et al. GraphDrawingBenchmark. https://github.com/yolandalalala/GraphDrawingBenchmark

---

## Appendix: Detailed Architecture Specifications

### A.1 GNNPolicy (REINFORCE)

| Layer | Input | Output | Parameters |
|-------|-------|--------|------------|
| input_proj | [N, 3] | [N, 128] | 3×128 + 128 = 512 |
| GCNLayer × 3 | [N, 128] | [N, 128] | 128×128 + 128 = 16,512 each |
| node_head | [N, 128] | [N, 1] | 128×64 + 64×1 = 8,256 |
| delta_head | [N, 128] | [N, 2] | 128×64 + 64×2 = 8,320 |
| **Total** | | | **≈ 66K parameters** |

### A.2 GATv2Policy (PPO)

| Layer | Input | Output | Parameters (4 heads, head_dim=32) |
|-------|-------|--------|-----------------------------------|
| input_proj | [N, 3] | [N, 128] | 512 |
| W_L, W_R, W_V × 3 layers | [N, 128] | [N, 128] | 3 × (3 × 128×128) = 147,456 |
| att × 3 layers | [N, N, 4, 32] | [N, N, 4] | 3 × 4 × 32 = 384 |
| LayerNorm × 3 layers | [N, 128] | [N, 128] | 3 × 256 = 768 |
| node_head | [N, 128] | [N, 1] | 8,256 |
| delta_head | [N, 128] | [N, 2] | 8,320 |
| value_head | [128] | [1] | 128×64 + 64×1 = 8,256 |
| **Total** | | | **≈ 174K parameters** |

### A.3 Training Hyperparameters

| Parameter | REINFORCE (GNN-RL) | PPO (GATv2-PPO) |
|-----------|:-----------------:|:---------------:|
| Epochs | 2,000 | 2,000 |
| LR (Adam) | 3×10⁻⁴ | 3×10⁻⁴ |
| Hidden dim | 128 | 128 |
| GNN layers | 3 | 3 |
| Attention heads | — | 4 |
| Max steps/ep | 300 | 300 (adaptive) |
| Discount γ | 0.99 | 0.99 |
| Entropy coef | 0.01 | 0.01 |
| Nodes/step | 1 | 2 |
| α (soft reward) | 1.0 | 1.0 |
| β (hard reward) | 5.0 | 5.0 |
| PPO clip ε | — | 0.2 |
| K epochs | — | 4 |
| GAE λ | — | 0.95 |
| Grad clip | 1.0 | 0.5 |
| Curriculum | No | Yes (5 stages) |
| Step size | 15.0 | 15.0 |
| σ (action noise) | 0.5 | 0.5 |

### A.4 Full Per-Graph Results (101 Test Graphs)

PPO column = quick-eval results (N\_TRIALS=5, MAX\_STEPS=300); submitted coordinates (PPO\*, MAX\_TRIALS=50, MAX\_STEPS=500 + SA refinement) achieve avg. 23.80 crossings (−29.79% vs. neato). See `eval_full_new.log` for the full PPO\* per-graph breakdown.

| Graph | N | neato | sfdp | SmGD | SA | MLP | GNN | PPO |
|-------|:-:|:-----:|:----:|:----:|:--:|:---:|:---:|:---:|
| grafo10000.38 | 38 | 14 | 9 | 13 | 10 | 8 | 9 | **7** |
| grafo10001.32 | 32 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| grafo10002.40 | 40 | 4 | 4 | 2 | 4 | 4 | 4 | 4 |
| grafo10003.40 | 40 | 17 | 8 | 10 | 9 | 10 | 9 | **8** |
| grafo10004.32 | 32 | 4 | 4 | 5 | 3 | 3 | 3 | 3 |
| grafo10005.39 | 39 | 15 | 15 | 17 | 14 | 12 | 11 | **9** |
| grafo10006.98 | 98 | 136 | 100 | 133 | 127 | 115 | 115 | **106** |
| grafo10007.31 | 31 | 2 | 0 | 0 | 1 | 2 | 1 | 1 |
| grafo10008.42 | 42 | 30 | 28 | 29 | 25 | 25 | 24 | **22** |
| grafo10009.31 | 31 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| grafo10010.39 | 39 | 11 | 16 | 12 | 10 | 10 | 9 | 10 |
| grafo10011.31 | 31 | 2 | 1 | 2 | 2 | 2 | 1 | 1 |
| grafo10012.40 | 40 | 12 | 12 | 13 | 11 | 9 | 9 | **8** |
| grafo10013.31 | 31 | 2 | 1 | 1 | 1 | 2 | 1 | 1 |
| grafo10014.39 | 39 | 13 | 18 | 17 | 12 | 12 | 11 | 11 |
| grafo10015.39 | 39 | 3 | 5 | 5 | 3 | 3 | 3 | 3 |
| grafo10016.39 | 39 | 20 | 16 | 14 | 19 | 18 | 15 | 16 |
| grafo10017.96 | 96 | 113 | 118 | 110 | 111 | 101 | 101 | **99** |
| grafo10018.31 | 31 | 4 | 3 | 5 | 4 | 4 | 4 | **3** |
| grafo10019.38 | 38 | 2 | 2 | 3 | 2 | 2 | 1 | 2 |
| grafo10020.38 | 38 | 17 | 24 | 21 | 16 | 14 | 14 | **14** |
| grafo10021.39 | 39 | 10 | 9 | 8 | 8 | 6 | 8 | **6** |
| grafo10022.31 | 31 | 2 | 1 | 1 | 1 | 2 | 1 | 1 |
| grafo10023.39 | 39 | 9 | 14 | 9 | 9 | 9 | 8 | 8 |
| grafo10024.32 | 32 | 5 | 3 | 5 | 5 | 5 | 3 | 4 |
| grafo10025.31 | 31 | 4 | 6 | 5 | 4 | 3 | 3 | 3 |
| grafo10026.32 | 32 | 4 | 5 | 6 | 3 | 3 | 3 | 3 |
| grafo10027.38 | 38 | 14 | 9 | 10 | 13 | 12 | 11 | 12 |
| grafo10028.96 | 96 | 104 | 111 | 106 | 95 | 93 | 89 | **86** |
| grafo10029.40 | 40 | 18 | 17 | 16 | 17 | 14 | 14 | **14** |
| grafo10030.38 | 38 | 14 | 13 | 14 | 13 | 12 | 11 | **11** |
| grafo10031.38 | 38 | 12 | 13 | 14 | 10 | 10 | 10 | **9** |
| grafo10032.38 | 38 | 9 | 9 | 9 | 6 | 7 | 7 | 7 |
| grafo10033.31 | 31 | 2 | 1 | 2 | 1 | 2 | 1 | 1 |
| grafo10034.40 | 40 | 18 | 13 | 16 | 16 | 16 | 16 | **15** |
| grafo10035.38 | 38 | 7 | 6 | 6 | 6 | 4 | 5 | **4** |
| grafo10036.31 | 31 | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| grafo10037.39 | 39 | 14 | 14 | 12 | 10 | 9 | 8 | 10 |
| grafo10038.38 | 38 | 7 | 6 | 8 | 6 | 6 | 5 | **5** |
| grafo10039.98 | 98 | 80 | 65 | 72 | 71 | 69 | 66 | **60** |
| grafo10040.32 | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| grafo10041.41 | 41 | 19 | 11 | 16 | 16 | 14 | 12 | **11** |
| grafo10042.39 | 39 | 14 | 11 | 17 | 14 | 11 | 10 | **10** |
| grafo10043.32 | 32 | 8 | 6 | 5 | 7 | 8 | 7 | **6** |
| grafo10044.38 | 38 | 10 | 9 | 11 | 10 | 9 | 6 | **6** |
| grafo10045.38 | 38 | 9 | 10 | 8 | 8 | 8 | 7 | 7 |
| grafo10046.40 | 40 | 7 | 11 | 8 | 7 | 6 | 6 | **6** |
| grafo10047.32 | 32 | 4 | 2 | 4 | 3 | 2 | 2 | **2** |
| grafo10048.32 | 32 | 5 | 8 | 2 | 5 | 5 | 4 | 4 |
| grafo10049.39 | 39 | 9 | 9 | 8 | 8 | 9 | 7 | 7 |
| grafo10050.97 | 97 | 117 | 105 | 118 | 113 | 101 | 99 | **99** |
| grafo10051.30 | 30 | 4 | 3 | 3 | 3 | 3 | 3 | 3 |
| grafo10052.32 | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| grafo10053.32 | 32 | 4 | 4 | 7 | 4 | 3 | 3 | **3** |
| grafo10054.40 | 40 | 13 | 14 | 20 | 12 | 11 | 11 | **10** |
| grafo10055.40 | 40 | 23 | 17 | 14 | 20 | 18 | 17 | **16** |
| grafo10056.31 | 31 | 2 | 1 | 5 | 1 | 1 | 1 | 1 |
| grafo10057.40 | 40 | 39 | 33 | 32 | 38 | 34 | 33 | **33** |
| grafo10058.32 | 32 | 3 | 10 | 4 | 3 | 3 | 3 | 3 |
| grafo10059.32 | 32 | 3 | 3 | 3 | 3 | 3 | 3 | **2** |
| grafo10060.90 | 90 | 50 | 48 | 54 | 47 | 43 | 42 | **42** |
| grafo10061.99 | 99 | 103 | 93 | 83 | 99 | 95 | 91 | **92** |
| grafo10062.39 | 39 | 6 | 5 | 4 | 6 | 5 | 3 | 4 |
| grafo10063.31 | 31 | 2 | 1 | 4 | 1 | 1 | 1 | 2 |
| grafo10064.39 | 39 | 26 | 27 | 27 | 23 | 21 | 19 | **18** |
| grafo10065.32 | 32 | 3 | 3 | 6 | 3 | 3 | 3 | 3 |
| grafo10066.40 | 40 | 19 | 23 | 22 | 15 | 15 | 15 | **13** |
| grafo10067.41 | 41 | 5 | 1 | 2 | 5 | 5 | 4 | 4 |
| grafo10068.31 | 31 | 2 | 1 | 2 | 1 | 2 | 1 | 1 |
| grafo10069.40 | 40 | 14 | 17 | 10 | 12 | 10 | 9 | **9** |
| grafo10070.32 | 32 | 1 | 5 | 1 | 1 | 1 | 1 | 1 |
| grafo10071.40 | 40 | 11 | 16 | 17 | 11 | 9 | 9 | **9** |
| grafo10072.97 | 97 | 110 | 95 | 118 | 107 | 96 | 92 | **86** |
| grafo10074.90 | 90 | 153 | 141 | 130 | 143 | 134 | 125 | **125** |
| grafo10075.31 | 31 | 1 | 2 | 2 | 1 | 1 | 1 | 1 |
| grafo10076.90 | 90 | 141 | 130 | 126 | 131 | 124 | 118 | **113** |
| grafo10077.31 | 31 | 2 | 1 | 1 | 1 | 2 | 1 | 1 |
| grafo10078.40 | 40 | 9 | 10 | 14 | 9 | 8 | 8 | **7** |
| grafo10079.41 | 41 | 25 | 22 | 24 | 20 | 19 | 18 | **18** |
| grafo10080.38 | 38 | 13 | 8 | 10 | 12 | 10 | 11 | **9** |
| grafo10081.34 | 34 | 9 | 10 | 7 | 8 | 7 | 6 | **5** |
| grafo10082.98 | 98 | 67 | 49 | 64 | 61 | 60 | 60 | **57** |
| grafo10083.38 | 38 | 11 | 18 | 10 | 10 | 9 | 8 | 9 |
| grafo10084.97 | 97 | 182 | 184 | 191 | 181 | 168 | 164 | **162** |
| grafo10085.35 | 35 | 7 | 5 | 4 | 7 | 4 | 5 | 4 |
| grafo10086.34 | 34 | 1 | 1 | 1 | 0 | 0 | 0 | **0** |
| grafo10087.39 | 39 | 12 | 10 | 11 | 10 | 8 | 7 | **7** |
| grafo10088.39 | 39 | 17 | 16 | 13 | 14 | 14 | 14 | 14 |
| grafo10089.37 | 37 | 12 | 19 | 12 | 11 | 10 | 8 | 9 |
| grafo10090.97 | 97 | 30 | 35 | 31 | 30 | 25 | 25 | **23** |
| grafo10091.96 | 96 | 72 | 70 | 67 | 62 | 63 | 60 | **59** |
| grafo10092.97 | 97 | 91 | 121 | 95 | 91 | 81 | 80 | 81 |
| grafo10093.98 | 98 | 109 | 117 | 112 | 106 | 100 | 93 | 97 |
| grafo10095.93 | 93 | 53 | 50 | 61 | 52 | 50 | 49 | **45** |
| grafo10096.95 | 95 | 103 | 81 | 92 | 97 | 93 | 89 | 90 |
| grafo10097.95 | 95 | 47 | 51 | 47 | 46 | 41 | 39 | **37** |
| grafo10098.95 | 95 | 104 | 102 | 112 | 101 | 94 | 94 | **93** |
| grafo10099.94 | 94 | 149 | 179 | 165 | 141 | 137 | 133 | **133** |
| grafo10100.99 | 99 | 104 | 102 | 90 | 101 | 87 | 87 | **86** |
| grafo10101.92 | 92 | 102 | 106 | 109 | 97 | 93 | 83 | **83** |
| grafo10102.97 | 97 | 53 | 58 | 60 | 52 | 48 | 41 | **42** |
