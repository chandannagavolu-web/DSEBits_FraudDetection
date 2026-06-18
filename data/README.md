# Data

## IEEE-CIS Fraud Detection

- **Source:** https://www.kaggle.com/datasets/lnasiri007/ieeecis-fraud-detection
- **License:** Kaggle competition data, for research/educational use only.
- **Files used:** `train_transaction.csv`, `train_identity.csv`
  (Kaggle's `test_*` files have no public `isFraud` labels, so they are not
  used for model evaluation. A stratified train/val/test split is created
  from the labeled training data instead — see
  `src/data/ieee_cis.py::make_splits`.)

### Expected layout

```
data/
├── raw/ieee_cis/
│   ├── train_transaction.csv
│   ├── train_identity.csv
├── interim/ieee_cis/
│   └── merged_clean.parquet      # output of src/data/ieee_cis.py
├── processed/ieee_cis/
│   ├── tabular.parquet           # output of src/features/ieee_cis_features.py
│   └── graph.pt                  # output of src/graph/ieee_cis_graph.py
```

### Preprocessing notes

- Transaction (`train_transaction.csv`) and identity (`train_identity.csv`)
  tables are left-joined on `TransactionID`.
- Categorical columns (`ProductCD`, `card1`-`card6`, `addr1`, `addr2`,
  `P_emaildomain`, `R_emaildomain`, `M1`-`M9`, `DeviceType`, `DeviceInfo`,
  `id_12`-`id_38`) are label-encoded with an explicit `"missing"` category.
- High-cardinality identifier columns (`card1`, `addr1`, `P_emaildomain`,
  `DeviceInfo`) are also frequency-encoded for the tabular baselines.
- Numeric columns (`TransactionAmt`, `C1`-`C14`, `D1`-`D15`, `V1`-`V339`,
  `id_01`-`id_11`) have missing values imputed with `-999` (out-of-range
  sentinel, standard practice for tree-based models) and are
  standard-scaled for the GNN / linear baselines.
- `TransactionAmt` is additionally log1p-transformed.

### Graph construction

For the GNN track, a heterogeneous transaction graph is built
(`src/graph/ieee_cis_graph.py`):

- **Node types:** `transaction`, `card1`, `addr1`, `P_emaildomain`,
  `DeviceInfo` (the latter four are "entity" nodes shared across
  transactions).
- **Edges:** `transaction --linked_to--> entity` and the reverse
  `entity --rev_linked_to--> transaction`, connecting a transaction to
  each entity value it shares (e.g., same card, same billing address,
  same email domain, same device fingerprint). Missing entity values are
  *not* linked, to avoid creating a single giant hub node.
- **Transaction node features:** the scaled numeric + label-encoded
  categorical feature matrix from `src/features/ieee_cis_features.py`.
- **Entity node features:** `[log1p(degree), mean(transaction numeric
  features over linked transactions)]` — derived only from features,
  never from the `isFraud` label, to avoid target leakage.
- **Labels / masks:** `isFraud` on `transaction` nodes, with
  `train_mask` / `val_mask` / `test_mask` from `make_splits`.
