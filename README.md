# Graph Layout Optimization with Reinforcement Learning

Minimize edge crossings in graph drawings using REINFORCE.

## Setup
pip install torch networkx pandas pygraphviz

## Data
Download Rome graph collection and place in `rome/`:
https://graphdrawing.unipg.it/data/rome-graphml.tgz

## Train
python train_all_sizes.py   # trains all node sizes
python train_only.py        # trains n=40 only (2000 epochs)

## Evaluate
python evaluate_all.py      # compare neato / sfdp / SA / RL on test set

## Results
| Method | Avg Crossings | vs neato |
|--------|--------------|---------|
| neato  | 29.08        | baseline |
| sfdp   | 28.86        | +0.79%   |
| SA     | 27.44        | +7.37%   |
| RL     | 25.09        | +14.54%  |
