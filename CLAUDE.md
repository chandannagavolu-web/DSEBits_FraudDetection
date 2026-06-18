# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

**Title:** Graph Neural Networks for Systemic Risk and Fraud Detection in Credit Systems
**Course:** DSECLZG628T – Dissertation (M.Tech, BITS Pilani WILP)
**Student:** Chandan Nagavolu 
**Research Area:** Credit Risk Management

This dissertation applies Graph Neural Networks (GNNs), Temporal Convolutional
Networks (TCNs), and Autoencoders to financial risk analysis — specifically
credit risk prediction and fraud/systemic-risk detection — and benchmarks
these deep learning approaches against traditional ML models.

## Objectives

1. Design and implement GNN models for detecting interconnected fraud and
   systemic risk (e.g., fraud rings, contagion across counterparties).
2. Develop temporal deep learning models (TCNs) for multi-period credit
   default prediction.
3. Build autoencoder-based anomaly detection systems for early warning signals.
4. Compare deep learning models (GNN/TCN/Autoencoder) against traditional ML
   baselines (Logistic Regression, XGBoost/LightGBM, Random Forest).
5. Integrate structured (tabular), relational (graph), and temporal (sequence)
   data into a unified predictive framework.

## Scope & Constraints

- Data sources: publicly available / simulated financial datasets only
  (no proprietary or real customer data, due to privacy constraints).
- Frameworks: PyTorch / TensorFlow, with PyTorch Geometric (PyG) or DGL for
  graph models.
- Evaluation must report both standard ML metrics and the credit
  risk / fraud detection industry-standard metrics (see below).
- Two parallel tracks:
  - **Fraud / systemic risk track** — graph-based, GNN-centric.
  - **Credit risk track** — tabular + temporal, TCN/Autoencoder-centric,
    scored with credit scoring industry metrics.

## Candidate Datasets

| Dataset | Track | Why it fits | Source |
|---|---|---|---|
| **Elliptic Bitcoin Dataset** | Fraud / systemic risk (graph-native) | ~200K nodes, 234K edges of Bitcoin transactions labeled licit/illicit/unknown — directly usable as a transaction graph for GNN node classification. | https://www.kaggle.com/datasets/ellipticco/elliptic-data-set |
| **IEEE-CIS Fraud Detection (Kaggle)** | Fraud (graph construction) | ~590K e-commerce transactions with identity/device fields; entities (card, device, email, address) can be linked into a heterogeneous transaction graph. | https://www.kaggle.com/datasets/lnasiri007/ieeecis-fraud-detection?select=test_transaction.csv |
| **PaySim (Synthetic Financial Dataset)** | Fraud / systemic risk | Simulated mobile money transfers between agents/customers/merchants — natural sender-receiver graph, useful for fraud-ring and money-laundering pattern detection. | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| **German Credit Data (Kaggle)** | Credit risk (tabular baseline) | Classic credit scoring dataset (1000 applicants, good/bad credit risk labels) — useful as a small, well-understood baseline for credit risk scorecard metrics (KS, Gini) before scaling to larger relational datasets. | https://www.kaggle.com/datasets/varunchawla30/german-credit-data |
| **Home Credit Default Risk (Kaggle)** | Credit risk (temporal + relational) | Multi-table relational data (application, bureau, previous applications, installment payments) — supports both temporal sequence modeling and entity-relationship graph construction for default prediction. | TBD |
| **Lending Club Loan Data** | Credit risk (temporal) | Loan-level data with repayment history over time — suited for TCN-based multi-period default prediction. | TBD |
| **Credit Card Fraud Detection (ULB/Kaggle)** | Fraud (tabular baseline) | ~284K anonymized transactions, highly imbalanced — useful as a tabular baseline / autoencoder anomaly-detection benchmark before graph extensions. | TBD |

Final dataset selection should be confirmed during the Literature Review /
Dissertation Outline phase and recorded in `data/README.md` along with
download source, license, and any preprocessing notes.

## Project Folder Structure

```
ProjectWork/
├── CLAUDE.md
├── README.md
├── Abstract.pdf
├── environment.yml / requirements.txt
├── data/
│   ├── raw/              # untouched downloaded datasets
│   ├── interim/          # cleaned but not yet feature-engineered
│   ├── processed/        # model-ready tensors/graphs/sequences
│   └── external/         # reference data, metadata, schemas
├── notebooks/
│   ├── 01_eda/                  # exploratory data analysis per dataset
│   ├── 02_graph_construction/   # building entity/transaction graphs
│   ├── 03_baseline_models/      # LogReg, XGBoost, RandomForest
│   ├── 04_gnn_models/           # GCN, GAT, GraphSAGE, etc.
│   ├── 05_temporal_models/      # TCN-based default prediction
│   └── 06_autoencoder/          # anomaly detection / early warning
├── src/
│   ├── data/             # loading, cleaning, splitting
│   ├── features/         # feature engineering
│   ├── graph/             # graph construction utilities (PyG/DGL)
│   ├── models/
│   │   ├── baseline/
│   │   ├── gnn/
│   │   ├── temporal/
│   │   └── autoencoder/
│   ├── evaluation/        # metrics implementations (see below)
│   └── utils/
├── configs/                # YAML/JSON experiment configs
├── models/                 # saved checkpoints / model artifacts
├── reports/
│   ├── figures/            # plots for dissertation
│   └── dissertation/       # write-up drafts, chapters
└── tests/                   # unit tests for src/
```

## Evaluation Metrics

In addition to standard metrics (Accuracy, Precision, Recall, F1-score,
ROC-AUC) called out in the Scope of Work, report the following
industry-standard metrics:

### Fraud Detection Track

- **PR-AUC (Precision-Recall AUC)** — area under the precision-recall curve;
  preferred over ROC-AUC for highly imbalanced fraud datasets since it
  focuses on performance on the positive (fraud) class.
- **Matthews Correlation Coefficient (MCC)** — a single balanced measure
  using all four confusion matrix categories (TP, TN, FP, FN); robust to
  class imbalance and ranges from -1 (total disagreement) to +1 (perfect
  prediction).
  ```
  MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
  ```

### Credit Risk Track

- **KS Statistic (Kolmogorov-Smirnov)** — maximum separation between the
  cumulative distributions of "good" and "bad" borrowers across predicted
  score deciles; a standard credit scorecard discrimination metric.
  Industry rule of thumb: KS > 40% is considered a strong model.
- **Gini Coefficient** — measures rank-ordering power of the model,
  computed as `Gini = 2*AUC - 1`; widely reported alongside AUC in credit
  scoring to communicate discriminatory power on a 0–1 scale familiar to
  risk teams.

All metric implementations should live in `src/evaluation/metrics.py` with
a shared interface so the same evaluation harness can score baseline,
GNN, temporal, and autoencoder models consistently.


## Literature References

- Cheng, D., Zou, Y., Xiang, S., Jiang, C. — *Review of Graph Neural Networks
  for Financial Fraud Detection*. https://arxiv.org/pdf/2411.05815
- Polu, O.R. et al. — *Graph Neural Networks for Fraud Detection: Modeling
  Financial Transaction Networks at Scale*.

## Working Conventions

- Use PyTorch + PyTorch Geometric for GNN work unless a specific paper's
  reference implementation requires DGL/TensorFlow.
- Keep notebooks for exploration only; Provide explanation of steps at each stage; reusable logic belongs in `src/`.
- Every experiment should log its config and metrics (PR-AUC, MCC, KS, Gini,
  plus standard metrics) to a results table for cross-model comparison in
  the dissertation's Testing phase.
- Ensure the project work is unique and doesn't have any plagiarism.
