# Graph Layout Optimization with Reinforcement Learning

Minimize edge crossings in graph drawings using REINFORCE on the Rome graph collection.

## Problem

Given an undirected graph, assign 2-D coordinates to each node to minimize the number of
edge crossings. Edge crossings are non-differentiable, so gradient descent cannot be
applied directly. This project uses Reinforcement Learning to learn a policy that
iteratively moves nodes to reduce crossings.

---

## Methods

### Baselines
| Method  | Description |
|---------|-------------|
| **neato**   | Graphviz spring-electrical layout (primary RL starting point) |
| **sfdp**    | Graphviz force-directed layout |
| **SmartGD** | Pretrained GNN from [GraphDrawingBenchmark](https://github.com/yolandalalala/GraphDrawingBenchmark) |
| **SA**      | Simulated Annealing from neato start (8 000 steps) |

### Our RL Methods
| Method     | Description |
|------------|-------------|
| **RL-MLP** | Per-node-size MLP policy trained with REINFORCE (one model per graph size) |
| **GNN-RL** | Size-agnostic GNN policy (GCN × 3) trained with REINFORCE on **all** sizes at once |

---

## Architecture

### RL-MLP (baseline RL)
- Flat MLP: `[N×2] → 256 → 256 → node_logits [N] + delta [N×2]`
- One model trained per node count N ∈ {31,32,34,35,37,38,39,40,41,42,90–99}
- Action: sample node → sample Gaussian displacement → move

### GNN-RL (proposed)
- **Node features**: `[x_norm, y_norm, degree_norm]` (3-dim)
- **Encoder**: 3 × residual GCN layers with LayerNorm, hidden dim = 128
- **Node selection head**: `[128 → 64 → 1]` → scores over all N nodes
- **Displacement head**: `[128 → 64 → 2]` → per-node `(dx, dy)` mean
- **Key advantage**: works for *any* graph size; trained on all sizes jointly

### Reward
```
r_t = alpha * (soft_crossing_before - soft_crossing_after)
    + beta  * (hard_crossing_before - hard_crossing_after)
```
`alpha=1, beta=5`. The soft crossing signal (sigmoid-smoothed) provides dense
learning signal even when the discrete count doesn't change.

### Episode
- Start from neato layout
- 120 steps per episode (GNN) / 200 steps (MLP)
- Track best layout seen; report its crossings
- Discounted returns (γ=0.99) with normalisation as variance-reduction baseline
- Entropy bonus (0.01) for exploration

---

## Results

Evaluation on Rome test set: **grafo10000 – grafo10100** (99 graphs).

Relative improvement vs neato (per graph, averaged):
```
rel_i = (method_i − neato_i) / max(method_i, neato_i)
mean_improvement = mean over test graphs  (negative = fewer crossings = better)
```

| Method      | Avg Crossings | vs neato       |
|-------------|:------------:|:--------------:|
| **neato**   | 29.57        | 0.00% (baseline) |
| sfdp        | 28.70        | −5.59%         |
| SmartGD     | 28.90        | −3.37%         |
| SA          | 27.61        | −12.90%        |
| RL-MLP      | 25.70        | −16.21%        |
| **GNN-RL**  | **24.95**    | **−21.05%**    |

**GNN-RL achieves −21.05% improvement over neato** (epoch-500 checkpoint), outperforming SmartGD (−3.37%), SA (−11.75%), and RL-MLP (−16.21%).

*(GNN training continues; run `python evaluate_full.py` after further training for updated GNN-RL numbers.)*

---

## Dataset

Rome graph collection: https://graphdrawing.unipg.it/data/rome-graphml.tgz

- **Training**: `grafo0001 – grafo9999` (up to 2 000 graphs, n ≤ 100 nodes)
- **Test**: `grafo10000 – grafo10100` (100 graphs)

Place all `.graphml` files in the `rome/` directory.

---

## Setup

```bash
pip install torch networkx pandas pygraphviz
```

> On Windows with conda: `conda install -c conda-forge pygraphviz`

---

## Reproduction

### 1 – Train RL-MLP (per-size models)
```bash
python train_all_sizes.py      # all node sizes, 1 000 epochs each
# or
python train_only.py           # n=40 only, 2 000 epochs
```
Pre-trained weights are already in `models/`.

### 2 – Train GNN-RL (size-agnostic)
```bash
python train_gnn.py            # 5 000 epochs, ~60 min on CPU
```
Saves `models/gnn_policy_final.pt` (checkpoints every 1 000 epochs).

### 3 – Evaluate all methods
```bash
python evaluate_full.py        # outputs eval_full.csv + console table
```

---

## File Structure

```
.
├── xing.py               # Edge crossing loss (hard + soft)
├── stress.py             # Stress layout loss
├── gnn_policy.py         # GNN policy architecture (NEW)
├── train_gnn.py          # GNN training with REINFORCE (NEW)
├── train_all_sizes.py    # Per-size MLP training
├── train_only.py         # Single-size MLP training (n=40)
├── evaluate_full.py      # Full 6-way evaluation (NEW)
├── evaluate_all.py       # 4-way evaluation (original)
├── metrics.csv           # SmartGD reference crossings
├── eval_full.csv         # Output: per-graph results (generated)
├── eval_all.csv          # Output: original 4-way results
├── models/
│   ├── gnn_policy_final.pt       # GNN model (generated)
│   ├── policy_n40_final.pt       # MLP for n=40
│   └── policy_n{N}.pt            # MLP for each size N
└── rome/                 # Rome .graphml files
```
