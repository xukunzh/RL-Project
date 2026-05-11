# Presentation Slides: Graph Layout Optimization via Reinforcement Learning

---

## Slide 1 — Title

**Graph Layout Optimization via Reinforcement Learning:**  
**Minimizing Edge Crossings**

CSA 5180 Final Project · Rome Graph Collection · April 2026

---

**Speaker Notes:**
Good morning. Today we present our final project on applying Reinforcement Learning to graph visualization — specifically, minimizing the number of edge crossings in graph drawings. This is a classical combinatorial problem with no known polynomial-time exact solution. We show that RL, with the right architecture and training recipe, outperforms state-of-the-art specialized deep learning systems. I'll walk you through the problem, our three-stage solution, four key improvements, and the full results on 101 test graphs.

---

## Slide 2 — Why Does Graph Layout Matter?

**The Problem at a Glance**

*[Visual: two drawings of the same graph — left with crossings, right cleaner]*

- Graph drawings appear everywhere: org charts, network maps, software dependencies, gene networks
- **Edge crossings** make graphs harder to read — they obscure which edges connect which nodes
- Classical methods (neato, sfdp) minimize **stress** — a smooth proxy — but **not crossings directly**
- Crossing count is **discrete and non-differentiable**: gradient descent cannot optimize it directly

> *Can we train an RL agent to discover node arrangements with fewer crossings?*

---

**Speaker Notes:**
Think of any visualization you encounter every day — a subway map, a flowchart, a family tree. When edges cross, the eye has to follow each edge carefully to figure out which node it connects to. Minimizing crossings directly improves readability. The challenge is that crossing count is a step function — it doesn't change continuously as you move nodes, so you can't take a gradient. Reinforcement learning is a natural fit because it can optimize non-differentiable, combinatorial objectives through trial-and-error search.

---

## Slide 3 — Problem Formulation: MDP

**Modeling Layout Optimization as a Markov Decision Process**

| MDP Component | Definition |
|---------------|-----------|
| **State** $s_t$ | Node coordinates $X_t \in \mathbb{R}^{N \times 2}$ + graph adjacency |
| **Action** $a_t$ | Select node(s); sample displacement $\Delta x \sim \mathcal{N}(\mu, \sigma^2 I)$ |
| **Reward** $r_t$ | $\alpha \cdot \Delta C_\text{soft} + \beta \cdot \Delta C_\text{hard}$ |
| **Initial state** | neato Graphviz layout (competitive starting point) |
| **Episode length** | Up to 300 steps; stop early if zero crossings |
| **Best tracking** | Report best layout seen during episode, not final |

**Key reward design:**
- **Hard reward** ($\beta = 5$): actual crossing reduction — sparse, only when integer count decreases
- **Soft reward** ($\alpha = 1$): sigmoid-smoothed crossing signal — dense, signal on every step
- Combined: stable training while still optimizing the true objective

---

**Speaker Notes:**
We model this as a Markov Decision Process. At each step, the agent sees the current node positions and the graph structure — that's the state. It picks one or two nodes to move and decides how far and in what direction — that's the action. The reward is based on how much the new layout reduces crossings. We use a combination of the real discrete count (hard reward) and a smoothed sigmoid approximation (soft reward). The soft reward is crucial: without it, an agent moving nodes in a dense graph may go hundreds of steps without ever changing the integer crossing count, resulting in zero learning signal.

---

## Slide 4 — Three Policy Architectures

**Our RL Progression**

### Architecture 1: RL-MLP
```
Input: [x₁, y₁, ..., xₙ, yₙ]  (flat coords, size 2N)
  ↓ Linear(256) → ReLU → Linear(256) → ReLU
  ↓                          ↓
node logits [N]          delta means [N×2]
```
- One model per graph size — **cannot generalize** across sizes
- Trained with **REINFORCE**

### Architecture 2: GNN-RL
```
Node features: [x_norm, y_norm, degree_norm]  (size 3, any N)
  ↓ 3 × GCN layer  (hidden=128, residual, LayerNorm)
  ↓                          ↓
node logits [N]          delta means [N×2]
```
- **One model for ALL graph sizes** — size-agnostic via graph convolution
- Trained with **REINFORCE**

### Architecture 3: GATv2-PPO (Our Best)
```
Same as GNN-RL, but:
  - GCN → GATv2 (4-head dynamic attention)
  - Added value head V(s) for PPO critic
  - k=2 multi-node actions per step
  - Curriculum learning over 5 size stages
```
- Trained with **PPO + GAE**

---

**Speaker Notes:**
We explored three architectures in sequence. First, the MLP: simple and fast, but requires a separate model for every graph size. Second, the GNN: one model for all sizes by using graph convolution — each node's features are aggregated from its neighbors through message passing. Third, our best system, GATv2-PPO: it replaces the fixed GCN weights with dynamic attention, uses PPO instead of REINFORCE for more stable training, moves two nodes per step, and uses curriculum learning.

---

## Slide 5 — REINFORCE vs PPO

**Why PPO Is Better for This Problem**

### REINFORCE (GNN-RL, RL-MLP)
- Collect episode → compute discounted returns $G_t = \sum_k \gamma^k r_{t+k}$
- Update: $\nabla \mathcal{L} = -\sum_t \log \pi(a_t|s_t) \cdot \hat{G}_t$
- ❌ High variance — reward signal is small and noisy (0–3 crossings per step)
- ❌ Each episode used only once — low sample efficiency

### PPO (GATv2-PPO)
- Compute **GAE advantages**: $\hat{A}_t = \sum_\ell (\gamma\lambda)^\ell (r_t + \gamma V(s_{t+1}) - V(s_t))$
- **Reuse each episode 4 times** with clipped surrogate:
$$\mathcal{L}^\text{CLIP} = \mathbb{E}_t\!\left[\min\!\left(\frac{\pi_\text{new}}{\pi_\text{old}} \hat{A}_t,\ \text{clip}\!\left(\frac{\pi_\text{new}}{\pi_\text{old}}, 0.8, 1.2\right)\hat{A}_t\right)\right]$$
- ✅ Value baseline $V(s)$ reduces variance substantially
- ✅ Clipping prevents catastrophic policy updates
- ✅ 4× sample efficiency from episode reuse

---

**Speaker Notes:**
REINFORCE is the classic policy gradient algorithm, but it has high variance. In our task, the reward signal is small and noisy — often just 0 or 1 crossing reduction per step. With high-variance gradients, the policy can easily get worse before it gets better. PPO addresses this with two mechanisms: first, a value function baseline that subtracts the expected future return, dramatically reducing variance. Second, the clipping constraint that prevents the policy from taking too large a step away from the old policy, even if a single episode suggests it should. We also reuse each episode 4 times, getting 4x the gradient updates per collected episode.

---

## Slide 6 — GATv2 + Multi-Node Actions

**Two More Improvements in GATv2-PPO**

### GCN (Fixed Weights) — Used in GNN-RL
$$h_i' = \text{ReLU}\!\left(\text{LN}\!\left(\sum_j \frac{1}{\sqrt{d_i d_j}} W h_j\right)\right)$$
- Neighbor $j$ always contributes proportionally to $1/\sqrt{d_i d_j}$ — **fixed**, depends only on degree

### GATv2 (Dynamic Attention) — Used in GATv2-PPO
$$\alpha_{ij} = \text{softmax}_j\!\left(\mathbf{a}^\top \text{LeakyReLU}(W_L h_i + W_R h_j)\right)$$
$$h_i' = \text{ReLU}\!\left(\text{LN}\!\left(\sum_j \alpha_{ij} W_V h_j\right)\right)$$
- Attention $\alpha_{ij}$ **depends on current node features** → focuses on most relevant neighbors

### Multi-Node Actions ($k = 2$)

| Single-node | Multi-node |
|:-----------:|:----------:|
| Move 1 node per step | Move 2 nodes jointly per step |
| Can get stuck when crossing requires both sides to move | Resolves coordinated crossings |
| 300 steps | Equivalent to 600 single-node opportunities |

---

**Speaker Notes:**
The third improvement is replacing the graph convolution layer with Graph Attention Network v2. GCN uses a fixed weight for each neighbor based only on degree. GATv2 computes a learned attention weight for each neighbor, based on both the current features of the node and the neighbor. This means the model can focus on the neighbors most informative about which direction to move, changing attention based on the current layout state. The fourth improvement, multi-node actions, addresses a fundamental limitation: when two edges cross, you often need to move both pairs of endpoints. Single-node moves can get trapped trying to resolve such crossings. By selecting two nodes per step, the agent can make coordinated improvements.

---

## Slide 7 — Curriculum Learning

**Training Small to Large**

```
Epoch progress:  0%      15%      30%      50%      75%     100%
                  |        |        |        |        |        |
Max graph size:  N≤30    N≤40    N≤55    N≤75    N≤100   N≤100
```

**Why this helps:**

| Without Curriculum | With Curriculum |
|:-----------------:|:---------------:|
| Large graphs (N=90) presented immediately | Small graphs (N≤30) presented first |
| Random policy: ~95% steps have zero reward | Short episodes, clear reward signal |
| Gradient noise overwhelms learning signal | Policy learns basic skills → scales up |
| Slower convergence, worse generalization | Faster convergence, better final result |

**Analogy:** You learn to ride a bike on flat ground before trying hills.

---

**Speaker Notes:**
The fifth and final improvement is curriculum learning. The intuition is simple: you learn to ride a bike on flat ground before trying hills. Similarly, our RL agent should learn basic node-moving strategies on small, sparse graphs before being confronted with dense graphs that have hundreds of crossings. In our implementation, the first 15% of training uses only graphs with at most 30 nodes. Each stage introduces larger graphs. By the time the agent sees the hardest graphs, it already has a solid policy that can quickly identify which nodes to move and in roughly which direction. This dramatically reduces gradient noise in early training.

---

## Slide 8 — Results on 101 Test Graphs

**Official Evaluation Metric:**
$$\text{Improvement} = \frac{1}{|\mathcal{T}|} \sum_{g \in \mathcal{T}} \frac{C_g^{\text{method}} - C_g^{\text{neato}}}{\max(C_g^{\text{method}},\ C_g^{\text{neato}})} \quad \text{(negative = better than neato)}$$

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

*Quick eval: N\_TRIALS=5; Submitted: N\_TRIALS=50, MAX\_STEPS=500 + SA refinement — no retraining.*

**Key takeaways:**
- Our methods (RL-MLP, GNN-RL, GATv2-PPO) all outperform SmartGD, a specialized pre-trained GNN
- GATv2-PPO submitted achieves **−29.79%** — nearly 30% better than neato
- Each RL generation improves on the last: MLP → GNN → PPO reflects systematic progress
- More test-time search (same model, more rollouts) adds another 6 pp with zero retraining

---

**Speaker Notes:**
Here are our main results on 101 test graphs. Neato is the baseline at 30.51 average crossings. Classical methods give 3–5% improvement. Simulated Annealing achieves 12%. Our per-size MLP policy achieves 15%, and the size-agnostic GNN achieves 23%. GATv2-PPO with standard evaluation achieves 24%, and with increased test-time search budget — 50 trials, 500 steps per trial, plus SA refinement — the submitted coordinates achieve nearly 30% improvement. No additional training was needed; the same model, given more search time, finds significantly better solutions. What's also remarkable is the comparison with SmartGD: all three of our RL methods surpass it, demonstrating that directly optimizing the crossing objective with RL is a powerful approach.

---

## Slide 9 — Visual Comparison

**Layout Visualization: Neato vs. GATv2-PPO**

*[Show viz_output/ side-by-side images: red edges = crossings, orange stars = crossing points]*

**Example: grafo10064 (N=39, 59 edges)**
- neato: 26 crossings — tangled central region
- GATv2-PPO (submitted): **17 crossings** — more spread out, clearer cluster structure (−35%)

**Large graph generalization: grafo10084 (N=97)**
- neato: 182 crossings
- GATv2-PPO (submitted): **154 crossings** (**−15%**, largest absolute reduction: −28)
- Model trained on N≤100 graphs generalizes to unseen large graphs

**Notable result: grafo10086 (N=34)**
- neato: 1 crossing → GATv2-PPO: **0 crossings** (perfect planar embedding found)

**Observation:** The RL policy learns to identify and untangle crossing-prone subregions, not just make random perturbations. Extended test-time search further amplifies these gains.

---

**Speaker Notes:**
Let's look at some actual drawings. The visualization shows side-by-side Neato vs. GATv2-PPO layouts, with crossing edges highlighted in red and crossing points marked with orange stars. For this 39-node graph, the neato layout has 26 crossings — you can see a tangle in the middle. The submitted GATv2-PPO layout has 17 crossings, a 35% reduction. For the large 97-node graph grafo10084 — the hardest in our test set — GATv2-PPO reduces crossings from 182 to 154, the largest absolute reduction of 28 crossings. And in one notable case — grafo10086 — the model finds a completely planar embedding with zero crossings.

---

## Slide 10 — Summary

**What We Built**

| Component | Detail |
|-----------|--------|
| MDP | Node displacement, soft+hard reward, best-tracking |
| RL-MLP | Per-size MLP + REINFORCE — **−15.49% vs neato** |
| GNN-RL | Size-agnostic GCN + REINFORCE — **−23.29% vs neato** |
| GATv2-PPO *(quick eval)* | GATv2 + PPO + multi-node + curriculum — **−23.86% vs neato** |
| **GATv2-PPO** *(submitted)* | +50 trials, 500 steps, SA refinement — **−29.79% vs neato** |

**Four training enhancements and their roles:**

| Enhancement | Addresses |
|-------------|-----------|
| **PPO** | High gradient variance; low sample efficiency of REINFORCE |
| **GATv2** | Fixed neighbor weights in GCN; context-insensitive aggregation |
| **Multi-node** | Single moves trapped on crossings requiring joint movement |
| **Curriculum** | Gradient noise on large graphs early in training |

**Main result: −29.79% vs neato** on 101 Rome test graphs (submitted), beating SmartGD (−3.12%), SA (−12.04%), and all other baselines. An additional +6 pp gained purely from test-time search, with no retraining.

---

**Speaker Notes:**
To summarize: we showed that RL is a powerful and competitive approach for minimizing edge crossings in graph layouts. The key insight is structural inductive bias — a GNN can identify crossing-prone nodes through message passing, and this representation generalizes across graph sizes. Our four training enhancements each address a concrete shortcoming. On top of that, we found that increasing the test-time search budget — more rollout trials, longer episodes, and SA refinement — pushes the submitted result to nearly 30% improvement over neato, with no additional training. This is analogous to best-of-N sampling in language models: a better policy makes more of each additional compute token. Thank you — I'm happy to take questions.

---

## Appendix A — GATv2 vs GCN Architecture Details

**GCN (Kipf & Welling, 2017)**
$$h' = \text{ReLU}(\text{LN}(W \cdot D^{-1/2}(A+I)D^{-1/2} \cdot h))$$
- Weight $\hat{A}_{ij} = 1/\sqrt{d_i d_j}$ — fixed for each graph topology, depends only on degree

**GATv2 (Brody et al., 2022)**
$$e_{ij} = \mathbf{a}^\top \text{LeakyReLU}(W_L h_i + W_R h_j)$$
$$\alpha_{ij} = \text{softmax}_j(e_{ij}), \quad h'_i = \text{ReLU}(\text{LN}(\textstyle\sum_j \alpha_{ij} W_V h_j))$$
- 4 attention heads, head dimension 32
- **Key over GAT v1:** LeakyReLU applied *after* combining $h_i$ and $h_j$ → full expressive power
- Attention adapts to current layout state at each step

---

## Appendix B — PPO Full Loss

$$\mathcal{L}(\theta) = \underbrace{-\mathbb{E}_t[\min(r_t \hat{A}_t, \text{clip}(r_t, 0.8, 1.2)\hat{A}_t)]}_{L^\text{CLIP}} + \underbrace{0.5 \cdot \mathbb{E}_t[(V_\theta(s_t) - R_t)^2]}_{L^\text{VF}} - \underbrace{0.01 \cdot H(\pi_\theta)}_{L^\text{ENT}}$$

- $r_t = \pi_\theta / \pi_{\theta_\text{old}}$ — probability ratio
- $R_t = \hat{A}_t + V(s_t)$ — return target for value head
- Gradient clipping: $\|\nabla\theta\|_2 \leq 0.5$, K=4 epochs per episode

---

## Appendix C — Hyperparameter Comparison

| Parameter | REINFORCE (GNN-RL) | PPO (GATv2-PPO) |
|-----------|:-----------------:|:---------------:|
| Epochs | 2,000 | 2,000 |
| Adam LR | 3×10⁻⁴ | 3×10⁻⁴ |
| Hidden | 128 | 128 |
| GNN layers | 3 | 3 |
| Attention heads | — | 4 |
| Max steps/ep | 300 | 300 (adaptive) |
| Nodes/step | 1 | 2 |
| Discount γ | 0.99 | 0.99 |
| Entropy coef | 0.01 | 0.01 |
| PPO clip ε | — | 0.2 |
| K epochs | — | 4 |
| GAE λ | — | 0.95 |
| Curriculum | No | Yes (5 stages) |
| Grad clip | 1.0 | 0.5 |
| Step size | 15.0 | 15.0 |
| σ (noise) | 0.5 | 0.5 |

---

## Appendix D — Selected Per-Graph Results

PPO\* = submitted coordinates (MAX\_TRIALS=50, MAX\_STEPS=500 + SA refinement).

| Graph | N | neato | sfdp | SmGD | SA | MLP | GNN | PPO\* |
|-------|:-:|:-----:|:----:|:----:|:--:|:---:|:---:|:-----:|
| grafo10008 | 42 | 30 | 28 | 29 | 25 | 25 | 24 | **19** |
| grafo10028 | 96 | 104 | 111 | 106 | 95 | 93 | 89 | **80** |
| grafo10039 | 98 | 80 | 65 | 72 | 71 | 69 | 66 | **58** |
| grafo10061 | 99 | 103 | 93 | 83 | 99 | 95 | 91 | **84** |
| grafo10064 | 39 | 26 | 27 | 27 | 23 | 21 | 19 | **17** |
| grafo10084 | 97 | 182 | 184 | 191 | 181 | 168 | 164 | **154** |
| grafo10086 | 34 | 1 | 1 | 1 | 0 | 0 | 0 | **0** |
| grafo10096 | 95 | 103 | 81 | 92 | 97 | 93 | 89 | **79** |
| grafo10099 | 94 | 149 | 179 | 165 | 141 | 137 | 133 | **124** |
| grafo10101 | 92 | 102 | 106 | 109 | 97 | 93 | 83 | **82** |
