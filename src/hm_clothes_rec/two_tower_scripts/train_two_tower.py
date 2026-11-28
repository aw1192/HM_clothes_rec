'''
Train the two tower retrieval model on train_retrieval, validate on train_rank.

usage: uv run python -m hm_clothes_rec.two_tower_scripts.train_two_tower
set TEST_RUN = False for the full run
'''

import duckdb
import torch

from hm_clothes_rec.dataset import TransactionDataset
from hm_clothes_rec.models.two_tower_model import TwoTowerRetrieval
from hm_clothes_rec.paths import ROOT, SPLITS_DIR

# test run: last 4 weeks of train_retrieval, 1 epoch
TEST_RUN = False

# mkdir for results and model weights, at the repo root no matter where the script is launched from
MODEL_DIR = ROOT / 'models'
RESULTS_DIR = ROOT / 'results'
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# initialize two tower model
tt = TwoTowerRetrieval()

# get the training data
train_path = SPLITS_DIR / 'train_retrieval.parquet'
if TEST_RUN:
    # last 4 weeks only
    train_df = duckdb.sql(f'''
        SELECT * FROM '{train_path}'
        WHERE t_dat >= (SELECT max(t_dat) FROM '{train_path}') - INTERVAL 27 DAY
    ''').df()
    train_dataset = TransactionDataset(train_df)
else:
    train_dataset = TransactionDataset(train_path)

# get the validation data: the raw train_rank purchases
val_dataset = TransactionDataset(SPLITS_DIR / 'train_rank.parquet')

print(f'train pairs {len(train_dataset):,} | val pairs {len(val_dataset):,} | cuda {torch.cuda.is_available()}')

# train the two-tower model
history = tt.fit(train_data=train_dataset,
                 val_data=val_dataset,
                 epochs=1 if TEST_RUN else 10)

# save the model weights and the per epoch losses / recall
name = 'two_tower_test' if TEST_RUN else 'two_tower'
tt.save(MODEL_DIR / f'{name}.pt')
torch.save(history, RESULTS_DIR / f'{name}_history.pt')
print(f'saved {MODEL_DIR / f"{name}.pt"} and {RESULTS_DIR / f"{name}_history.pt"}')
