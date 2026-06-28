import urllib.request
from pathlib import Path

import pandas as pd

from src.data.german_credit import load_raw, make_splits
from src.evaluation.evaluate_german_credit_ks_gini import run as evaluate_german_credit_ks_gini
from src.features.german_credit_features import (
    GermanCreditFeaturePipeline,
    information_value_summary,
)
from src.models.baseline.train_baseline_german_credit import run as train_baseline_german_credit
from src.utils.io import load_config

cfg = load_config('configs/german_credit.yaml')
raw_dir = Path(cfg['paths']['raw_dir'])
raw_dir.mkdir(parents=True, exist_ok=True)
raw_path = raw_dir / cfg['paths']['data_file']

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
with urllib.request.urlopen(url, timeout=20) as resp:
    text = resp.read().decode('utf-8', 'ignore')

cols = [
    'Status of existing checking account', 'Duration in month', 'Credit history', 'Purpose',
    'Credit amount', 'Savings account/bonds', 'Present employment since', 'Installment rate in percentage of disposable income',
    'Personal status and sex', 'Other debtors / guarantors', 'Present residence since', 'Property',
    'Age in years', 'Other installment plans', 'Housing', 'Number of existing credits at this bank',
    'Job', 'Number of people being liable to provide maintenance for', 'Telephone', 'foreign worker', 'Risk'
]
df = pd.read_csv(pd.io.common.StringIO(text), sep=' ', header=None, names=cols)
df = df.loc[:, ~df.columns.duplicated()]
df['Risk'] = (df['Risk'] == 2).astype(int)
df = df.rename(columns={
    'Status of existing checking account': 'Checking account',
    'Duration in month': 'Duration',
    'Savings account/bonds': 'Saving accounts',
    'Present employment since': 'Employment',
    'Personal status and sex': 'Sex',
    'Other debtors / guarantors': 'Guarantors',
    'Present residence since': 'Residence',
    'Age in years': 'Age',
    'Other installment plans': 'Installment plans',
    'Number of existing credits at this bank': 'Existing credits',
    'Number of people being liable to provide maintenance for': 'Liable people',
})
df['Checking account'] = df['Checking account'].map({
    'A11': 'little', 'A12': 'moderate', 'A13': 'rich', 'A14': 'unknown'
})
df['Purpose'] = df['Purpose'].map({
    'A40': 'car', 'A41': 'radio/TV', 'A42': 'education', 'A43': 'furniture/equipment',
    'A44': 'household', 'A45': 'repair', 'A46': 'education', 'A47': 'vacation/others',
    'A48': 'retraining', 'A49': 'business', 'A410': 'others'
})
df['Credit history'] = df['Credit history'].map({
    'A30': 'no credits', 'A31': 'all paid', 'A32': 'existing paid',
    'A33': 'delayed previously', 'A34': 'critical'
})
df['Saving accounts'] = df['Saving accounts'].map({
    'A61': 'little', 'A62': 'moderate', 'A63': 'rich', 'A64': 'quite rich', 'A65': 'unknown'
})
df['Sex'] = df['Sex'].map({'A91': 'male', 'A92': 'female', 'A93': 'male', 'A94': 'male'})
df['Housing'] = df['Housing'].map({'A151': 'rent', 'A152': 'own', 'A153': 'free'})
df['Job'] = df['Job'].map({0: 'unskilled', 1: 'skilled', 2: 'skilled', 3: 'highly skilled'})

for col in ['Sex', 'Job', 'Housing', 'Purpose', 'Checking account', 'Saving accounts', 'Credit history']:
    df[col] = df[col].astype(str)

keep_cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose', 'Risk']
df = df[keep_cols]
df.to_csv(raw_path, index=False)
print('wrote', raw_path)

df2 = load_raw(raw_dir, cfg['paths']['data_file'], cfg['features']['label_col'])
splits = make_splits(df2, label_col=cfg['split']['stratify_col'])
print('loaded rows', len(df2))
print('split sizes', {k: len(v) for k, v in splits.items()})

pipeline = GermanCreditFeaturePipeline(
    label_col=cfg['features']['label_col'],
    id_col=cfg['features']['id_col'],
    categorical_cols=cfg['features']['categorical_cols'],
    log1p_cols=cfg['features']['log1p_cols'],
    missing_value=cfg['features']['missing_value'],
)
train_feats = pipeline.fit_transform(splits['train'])
val_feats = pipeline.transform(splits['val'])
test_feats = pipeline.transform(splits['test'])
print('feature shapes', train_feats.shape, val_feats.shape, test_feats.shape)

iv_summary = information_value_summary(splits['train'], cfg['features']['iv_cols'], cfg['features']['label_col'])
print(iv_summary.head().to_string(index=False))

processed_dir = Path(cfg['paths']['processed_dir'])
processed_dir.mkdir(parents=True, exist_ok=True)
for name, feats in [('train', train_feats), ('val', val_feats), ('test', test_feats)]:
    (processed_dir / f'tabular_{name}.parquet').parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(processed_dir / f'tabular_{name}.parquet')
pipeline.save(processed_dir / 'feature_pipeline.pkl')

interim_dir = Path(cfg['paths']['interim_dir'])
interim_dir.mkdir(parents=True, exist_ok=True)
# The trainer's CV step expects the full, label-normalised data in interim/clean.parquet.
df2.to_parquet(interim_dir / 'clean.parquet')

train_baseline_german_credit(cfg)
evaluate_german_credit_ks_gini(cfg)
print('completed german credit workflow')
