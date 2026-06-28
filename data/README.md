# Data

## German Credit Data

- **Source:** https://www.kaggle.com/datasets/varunchawla30/german-credit-data
- **License:** Based on the UCI Statlog (German Credit Data) set (Prof.
  Hans Hofmann); open for research/educational use.
- **Track:** Credit risk (tabular baseline). This is the small, well-understood
  starting point for the credit-risk scorecard metrics (KS, Gini, WoE/IV)
  before scaling to larger relational datasets. There is **no graph and no
  GNN** here — the data is purely tabular.
- **File used:** a single CSV (~1000 applicants), placed manually under
  `data/raw/german_credit/`. The default expected name is
  `german_credit_data.csv` (override `paths.data_file` in
  `configs/german_credit.yaml` if yours differs).

### Two circulating schemas

The Kaggle file ships in one of two layouts; the loader
(`src/data/german_credit.py`) handles either target encoding automatically, but
the **feature column names in the config must match your file**:

- **Cleaned version (default this config targets):** columns `Age, Sex, Job,
  Housing, Saving accounts, Checking account, Credit amount, Duration, Purpose`
  plus a `Risk` column valued `good` / `bad`. Note `Saving accounts` and
  `Checking account` contain genuine `NaN`s. A leading `Unnamed: 0` export index
  is dropped on load.
- **UCI Statlog version:** 21 coded attributes (`A11`, `A34`, …) with the target
  coded `1` (good) / `2` (bad). If you downloaded this one, update the
  `features.*` lists in `configs/german_credit.yaml` to its column names.

### Expected layout

```
data/
├── raw/german_credit/
│   └── german_credit_data.csv
├── interim/german_credit/
│   └── clean.parquet                       # output of src/data/german_credit.py
├── processed/german_credit/
│   ├── tabular_{train,val,test}.parquet    # src/features/german_credit_features.py
│   └── feature_pipeline.pkl
```

### Preprocessing notes

- **Target convention:** normalised to **bad = 1, good = 0** regardless of the
  source encoding (good/bad strings or 1/2 codes), since "bad" (default) is the
  event KS / Gini measure separation for.
- No applicant id exists, so a synthetic `ApplicantID` (row position) is added
  to align feature rows with split membership.
- **Split:** stratified 70/15/15 train/val/test on `Risk`
  (`src/data/german_credit.py::make_splits`). No sub-sampling — the whole
  dataset fits in memory trivially.
- **Feature engineering** (`src/features/german_credit_features.py`,
  `GermanCreditFeaturePipeline`): mirrors the fraud-track pipelines — fit once on
  train, reused for val/test. `Job` (an ordinal skill code 0-3) and the other
  categoricals (`Sex`, `Housing`, `Saving accounts`, `Checking account`,
  `Purpose`) are label-encoded with an explicit `"missing"` category for the
  NaN-bearing account columns. Numeric columns (`Age`, `Credit amount`,
  `Duration`) get a sentinel-filled "raw" view (tree models) and a median-filled
  + standard-scaled view (logistic regression); `Credit amount` is additionally
  log1p-transformed. There is no frequency encoding (no high-cardinality ids).
- **Weight-of-Evidence / Information Value:** computed on the **train split
  only** over `features.iv_cols` and written to `reports/german_credit_iv.csv`
  for the dissertation's feature-screening section (IV bands: <0.02 useless,
  0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspicious/leakage).

### Run order

```
python -m src.data.german_credit                          # load + split -> interim
python -m src.features.german_credit_features             # features + IV -> processed
python -m src.models.baseline.train_baseline_german_credit  # baselines + 5-fold CV
python -m src.evaluation.evaluate_german_credit_ks_gini   # KS/Gini scorecard
```

Because the dataset is tiny (~1000 rows), a single 150-row test split gives a
noisy KS/Gini, so the trainer also reports **5-fold stratified CV** mean/std
(`split=cv_mean` / `cv_std` in `reports/results_german_credit.csv`), with the
feature pipeline refit inside each fold to avoid leakage.

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

## PaySim (Synthetic Financial Dataset)

- **Source:** https://www.kaggle.com/datasets/ealaxi/paysim1
- **License:** CC BY-SA 4.0 — synthetic data generated by the PaySim mobile-money
  simulator (Lopez-Rojas et al.), for research/educational use.
- **File used:** a single CSV, `PS_20174392719_1491204439457_log.csv`
  (~6.3M rows, ~470 MB). Download it manually from Kaggle and place it under
  `data/raw/paysim/` (or update `paths.transaction_file` in
  `configs/paysim.yaml` if you renamed it).

### Expected layout

```
data/
├── raw/paysim/
│   └── PS_20174392719_1491204439457_log.csv
├── interim/paysim/
│   └── clean.parquet               # output of src/data/paysim.py
├── processed/paysim/
│   ├── tabular_{train,val,test}.parquet  # src/features/paysim_features.py
│   ├── feature_pipeline.pkl
│   └── graph.pt                    # src/graph/paysim_graph.py
```

### Schema & preprocessing notes

- Columns: `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
  nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`.
  PaySim has no transaction id, so `src/data/paysim.py` adds a synthetic
  `TransactionID` (the row position) to align graph node masks with feature rows.
- **Sub-sampling:** the full transaction graph (~6.3M tx nodes + millions of
  account nodes) does not fit in 8 GB RAM. `configs/paysim.yaml::subsample`
  takes a *stratified* fraction (default `0.15`) **before** splitting, so the
  baselines and the GNN train/evaluate on identical rows (fair comparison) and
  the rare fraud rate (~0.13%) is preserved. Set `fraction: 1.0` on a larger
  machine.
- **Stratified split:** train/val/test (70/15/15) stratified on `isFraud`
  (`src/data/paysim.py::make_splits`); val/test keep the real fraud rate so
  PR-AUC / MCC / KS / Gini reflect real-world performance.
- **Engineered features** (`src/features/paysim_features.py`):
  `errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg` and
  `errorBalanceDest = oldbalanceDest + amount - newbalanceDest` (balance-
  consistency, the strongest PaySim fraud signal), signed balance deltas,
  `hour = step % 24`, `day = step // 24`, and `isMerchantDest` (PaySim
  merchant accounts are prefixed `"M"`).
- `type` is label-encoded; the high-cardinality account ids (`nameOrig`,
  `nameDest`) are frequency-encoded (transaction degree). Numeric columns get
  a sentinel-filled "raw" view (tree models) and a median-filled + scaled view
  (linear / GNN), with `amount` and the four balance columns log1p-transformed.
- **Leakage guard:** `isFlaggedFraud` (a downstream business rule) and the raw
  account-id strings are dropped from the model feature matrix (`drop_cols`).
  Note: in PaySim, fraud only ever occurs in `TRANSFER` / `CASH_OUT`
  transactions — this is kept as natural signal, not filtered out.

### Graph construction

For the GNN track (`src/graph/paysim_graph.py`) a heterogeneous graph is built:

- **Node types:** `transaction` (one per row) and a single `account` node type
  spanning **both** `nameOrig` and `nameDest`. Unifying the account vocabulary
  is essential: the same account can send in one transaction and receive in
  another, and merging them is what lets the GNN follow money across hops
  (fraud rings / layering). This differs from the IEEE-CIS builder, where each
  entity column became its own node type.
- **Edges:** `transaction --sends_from--> account` (origin) and
  `transaction --sends_to--> account` (destination), plus the reverse edges
  for bidirectional message passing.
- **Transaction node features:** the scaled feature matrix from
  `src/features/paysim_features.py`.
- **Account node features:** `[log1p(total_degree), mean(transaction features
  over all incident transactions)]` — derived only from features, never from
  the `isFraud` label, to avoid target leakage.
- **Labels / masks:** `isFraud` on `transaction` nodes, with
  `train_mask` / `val_mask` / `test_mask` from `make_splits`.

## Elliptic Bitcoin Dataset

- **Source:** https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- **License:** Elliptic, released for research/educational use (see the Kaggle
  page); anonymised — node ids and feature names are not human-readable.
- **Track:** Fraud / systemic risk (GNN node classification). Unlike IEEE-CIS and
  PaySim, this is a **homogeneous, temporal, pre-featurized** graph: one
  `transaction` node type, features already engineered, and the payment edges
  given directly. There is **no feature engineering and no entity-graph
  construction** — the graph is the dataset.
- **Files used:** three CSVs, placed manually under `data/raw/elliptic/`:
  - `elliptic_txs_features.csv` — **no header**, 167 columns: `txId`,
    `time_step` (1–49), then 165 anonymised features.
  - `elliptic_txs_classes.csv` — header `txId,class`, values `1` (illicit),
    `2` (licit), `unknown`.
  - `elliptic_txs_edgelist.csv` — header `txId1,txId2`, ~234K directed payment
    edges.

### Expected layout

```
data/
├── raw/elliptic/
│   ├── elliptic_txs_features.csv
│   ├── elliptic_txs_classes.csv
│   └── elliptic_txs_edgelist.csv
└── processed/elliptic/
    └── graph.pt                     # output of src/graph/elliptic_graph.py
```

### Schema & preprocessing notes

- **Label convention:** the 3-way `class` column is mapped to **illicit = 1,
  licit = 0, unknown = -1** (`src/data/elliptic.py`). About 21% of nodes are
  labelled (~2% illicit, ~19% licit) and ~77% are `unknown`. Unknown nodes are
  kept in the graph for message passing but excluded from the loss and every
  metric.
- **Feature blocks:** with the time step the 166-wide feature block splits into
  the first **94 local** features (time step + 93 transaction-local) and **72
  aggregated** one-hop neighbourhood roll-ups (Weber et al. 2019). The config's
  `features.feature_set` selects `local` (LF, 94) or `all` (AF, 166) for the
  AF/LF ablation; `drop_time_step_feature` optionally removes the raw clock.
- **Temporal (out-of-time) split** (`src/data/elliptic.py::make_temporal_masks`)
  over the 49 time steps — the canonical Elliptic protocol, replacing the random
  stratified split used by the other tracks. Defaults: train = steps 1–30,
  val = 31–34, test = 35–49 (`split.{val_min_step,train_max_step,test_min_step}`
  in `configs/elliptic.yaml`). Training never sees a future time step, so the
  evaluation is a realistic forward-in-time forecast.
- **Standardization** (`src/features/elliptic_features.py`): features are
  standardized using **train-node statistics only**, then applied to
  val/test/unknown nodes, so the temporal split is never leaked.

### Graph construction

For the GNN track (`src/graph/elliptic_graph.py`) a **homogeneous** PyG `Data`
object is built — the key structural difference from the IEEE-CIS / PaySim
heterogeneous builders:

- **Node type:** a single `transaction` node per row (no entity nodes).
- **Edges:** the dataset's directed `txId1 → txId2` payment flows; reverse edges
  are added (`gnn.add_reverse_edges`) so message passing is bidirectional.
- **Node features:** the standardized feature matrix (AF or LF per config).
- **Labels / masks:** `y` ∈ {1, 0, -1}, with `train_mask` / `val_mask` /
  `test_mask` / `labeled_mask` (boolean over all nodes).

### Models

- **Baselines** (`src/models/baseline/train_baseline_elliptic.py`): Logistic
  Regression, Random Forest, XGBoost trained on the *same* node feature matrix
  the GNN consumes (read straight out of `graph.pt`), ignoring the edges — so
  the gap versus the GNN measures what the payment-flow structure adds. On
  Elliptic, Random Forest is a famously strong baseline that rivals/beats GCN on
  illicit-class F1.
- **GNN** (`src/models/gnn/homo_gnn.py`, `train_gnn_elliptic.py`): a homogeneous
  GCN / GraphSAGE / GAT (`gnn.model_type`), full-batch transductive, loss masked
  to labelled train nodes, early-stopped on val PR-AUC.

### Run order

```
python -m src.graph.elliptic_graph                       # load + temporal split -> graph.pt
python -m src.models.baseline.train_baseline_elliptic    # LogReg / RF / XGBoost
python -m src.models.gnn.train_gnn_elliptic              # homogeneous GNN
```

Fraud-track metrics (PR-AUC, MCC, illicit-class F1) are the headline here; the
shared `compute_all_metrics` also reports KS/Gini for cross-track consistency.

## Lending Club Loan Data

- **Source:** Kaggle `wordsforthewise/lending-club`
  (`accepted_2007_to_2018Q4.csv`, ~2.26M accepted loans, ~151 columns).
- **License:** Lending Club loan-level data, public for research/educational use.
- **Track:** Credit risk (tabular **+ temporal**). This is the credit track's
  large, real-world dataset and the home of the **TCN** multi-period default
  model (CLAUDE.md Objective 2). Place the CSV under `data/raw/lending_club/`.

### Two concerns that define this track

1. **Leakage.** Lending Club is the canonical leakage trap: dozens of columns
   (`recoveries`, `total_pymnt*`, `last_pymnt_*`, `out_prncp*`, `last_fico_*`,
   the `hardship_*` / `settlement_*` blocks) record post-origination outcomes
   that would trivially reveal the label. The loader (`src/data/lending_club.py`)
   reads only an **allowlist** of origination-time columns (`load.usecols` in
   `configs/lending_club.yaml`) — a drop-list would be fragile on a 151-column
   file, an allowlist cannot leak by accident.
2. **Censoring.** Only terminal-status loans are kept: `Fully Paid` (good = 0)
   vs `Charged Off` / `Default` (bad = 1). In-progress statuses (`Current`,
   `Issued`, `In Grace Period`, `Late ...`) are dropped because their outcome is
   not yet known. (Documented bias: this favours matured vintages.)

### Expected layout

```
data/
├── raw/lending_club/
│   └── accepted_2007_to_2018Q4.csv
├── interim/lending_club/
│   └── clean.parquet                       # output of src/data/lending_club.py
└── processed/lending_club/
    ├── tabular_{train,val,test}.parquet    # src/features/lending_club_features.py
    └── feature_pipeline.pkl
```

### Preprocessing notes

- **String-coded fields parsed** in the loader: `term` (`"36 months"` → 36),
  `emp_length` (`"10+ years"` → 10, `"< 1 year"` → 0), `int_rate` / `revol_util`
  (strip `%`). **Engineered:** `fico` = midpoint of `fico_range_low/high`,
  `credit_age_mths` = months from `earliest_cr_line` to `issue_d`.
- **Out-of-time split** (`make_temporal_splits`): loans are ordered by `issue_d`
  and sliced 70/15/15 — earliest vintages train, latest vintages test — so the
  scorecard is always validated on issue periods it could not have seen. This
  replaces the random stratified split used by German Credit. **No k-fold CV**
  (it would shuffle vintages together and leak time).
- **Feature engineering** (`LendingClubFeaturePipeline`): same fit-on-train /
  reuse contract as the other tracks — raw (sentinel-filled) numeric view for
  tree models, median-filled + scaled view for the linear model / TCN;
  low-card categoricals label-encoded; high-card `addr_state` **frequency-
  encoded**; `annual_inc` / `revol_bal` / `installment` log1p-transformed. WoE/IV
  screening (train split only) is shared with the German Credit pipeline.

### Temporal model (TCN, Phase 2)

`src/models/temporal/tcn.py` is a dilated causal TCN (Bai et al. 2018); the
sequences come from `src/features/lending_club_sequences.py`.

- **Caveat — synthetic sequences.** The accepted-loans CSV is a one-row-per-loan
  *snapshot*, not a monthly panel. The default `sequences.mode:
  synthetic_amortization` manufactures each loan's monthly **amortization
  schedule** (scheduled balance / principal / interest trajectory) as the
  temporal axis and broadcasts the static origination features across the
  timesteps. This exercises the full TCN training/eval harness but is a
  **proof-of-concept** temporal signal (the schedule is deterministic from the
  origination terms). For genuine per-period sequences, obtain Lending Club's
  payment-history panel (`PMTHIST_ALL_*.csv`) and implement the
  `payment_history` branch in `lending_club_sequences.py`.

### Run order

```
python -m src.data.lending_club                            # load + clean + OOT split -> interim
python -m src.features.lending_club_features               # features + IV -> processed
python -m src.models.baseline.train_baseline_lending_club  # LogReg / RF / XGBoost
python -m src.evaluation.evaluate_lending_club_ks_gini     # KS/Gini scorecard
python -m src.models.temporal.train_tcn_lending_club       # TCN (Phase 2)
```

All five models log to `reports/results_lending_club.csv` for cross-model
comparison; KS/Gini are the credit-track headline.

## Credit Card Fraud Detection (ULB)

- **Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **License:** Open for research/educational use (Université Libre de Bruxelles,
  Machine Learning Group).
- **Track:** Fraud — **unsupervised autoencoder anomaly detection** (CLAUDE.md
  Objective 3), benchmarked against supervised baselines (Objective 4).
- **File used:** a single CSV `creditcard.csv` (~284,807 transactions), placed
  manually under `data/raw/creditcard/`.

### Schema

- 30 numeric columns: `Time` (seconds since first transaction), `V1`..`V28`
  (PCA-anonymised — the original features are confidential), `Amount`, plus a
  binary `Class` (**fraud = 1, ~0.172%**). There is no transaction id, so a
  synthetic `TxnID` (row position) is added (`src/data/creditcard.py`).

### Expected layout

```
data/
├── raw/creditcard/
│   └── creditcard.csv
└── processed/creditcard/
    ├── tabular_{train,val,test}.parquet   # src/features/creditcard_features.py
    └── feature_pipeline.pkl
```

### Approach & preprocessing notes

- **Split:** stratified 70/15/15 train/val/test on `Class` (this is a tabular
  anomaly-detection problem — no graph, no time axis).
- **Scaling** (`CreditCardFeaturePipeline`): `Time` is dropped by default
  (`features.drop_time`); the remaining 29 features (`V1`..`V28` + `Amount`) are
  standardized. The scaler is fit on the **train-normal rows only**, so the
  autoencoder learns normal-traffic geometry without the rare fraud cases (or the
  held-out splits) shifting the mean/variance. The same scaled matrix feeds the
  supervised baselines for a fair comparison.
- **Autoencoder** (`src/models/autoencoder/`): a symmetric MLP autoencoder is
  trained on **normal transactions only**; per-sample **reconstruction error**
  (MSE) is the anomaly score — fraud reconstructs poorly because it is unlike
  anything seen in training. The decision threshold is tuned on validation
  (`autoencoder.threshold_method`: `best_f1` sweep or a high `percentile` of the
  normal-error distribution). Ranking metrics (PR-AUC/ROC-AUC/KS/Gini) use the
  raw error; F1/MCC use the tuned threshold.
- **Baselines** (`src/models/baseline/train_baseline_creditcard.py`): supervised
  Logistic Regression / Random Forest / XGBoost on the same scaled features —
  the Objective-4 comparison (how close an *unsupervised* detector gets to
  *supervised* models that were shown labelled fraud).
- **Metrics:** at 0.172% fraud, **PR-AUC and MCC are the headline** (ROC-AUC is
  over-optimistic under this imbalance).

### Run order

```
python run_creditcard_pipeline.py      # load -> features -> baselines -> AE -> AE analysis
```

or step by step: `src.data.creditcard` → `src.models.baseline.train_baseline_creditcard`
→ `src.models.autoencoder.train_autoencoder_creditcard` → `src.evaluation.evaluate_creditcard`.
The AE analysis writes the reconstruction-error distribution, PR curve, and a
threshold sweep under `reports/figures/` and `reports/creditcard/`.
