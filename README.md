# Graph Neural Networks for Systemic Risk and Fraud Detection in Credit Systems

M.Tech Dissertation (DSECLZG628T, BITS Pilani WILP) — Chandan Nagavolu.

This repo applies GNNs, TCNs, and Autoencoders to credit risk and fraud /
systemic-risk detection, benchmarked against traditional ML baselines.
See [CLAUDE.md](CLAUDE.md) for the full project plan and conventions.

## 1. Setup

Requires Python 3.10+.

```bash
python -m venv .venv 
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # macOS/Linux

pip install -r requirements.txt
```

`torch` and `torch_geometric` versions in `requirements.txt` are unpinned;
if you need GPU support, install the CUDA-matched wheels from
https://pytorch.org and https://pytorch-geometric.readthedocs.io first,
then `pip install -r requirements.txt` for the rest.

## 2. Get the data (IEEE-CIS Fraud Detection)

Download `train_transaction.csv` and `train_identity.csv` from
[Kaggle: IEEE-CIS Fraud Detection](https://www.kaggle.com/datasets/lnasiri007/ieeecis-fraud-detection)
and place them at:

```
data/raw/ieee_cis/train_transaction.csv
data/raw/ieee_cis/train_identity.csv
```

See [data/README.md](data/README.md) for dataset details, preprocessing
notes, and the graph construction rationale.

## 3. Run the pipeline

Run each stage from the project root (so `configs/ieee_cis.yaml` resolves
correctly). Each step reads/writes the paths configured in
[configs/ieee_cis.yaml](configs/ieee_cis.yaml).

```bash
# 1. Load, merge transaction+identity tables, create train/val/test split
python -m src.data.ieee_cisturetu

# 2. Fit feature pipeline, write processed tabular splits + feature_pipeline.pkl
python -m src.features.ieee_cis_features

# 3. Build the heterogeneous transaction graph (graph.pt)
python -m src.graph.ieee_cis_graph

# 4. Train/evaluate baseline models (Logistic Regression, Random Forest, XGBoost)
python -m src.models.baseline.train_baseline

# 5. Train/evaluate the heterogeneous GNN
python -m src.models.gnn.train_gnn
```

Each run appends its metrics (Accuracy, Precision, Recall, F1, ROC-AUC,
PR-AUC, MCC) to `reports/results_ieee_cis.csv` for cross-model comparison.

## 4. Tests

```bash
pytest
```

## 5. Notebooks

Exploratory work lives under `notebooks/` (EDA, graph construction,
baselines, GNN, temporal, autoencoder). Reusable logic should live in
`src/` and be imported from notebooks, not duplicated.