'''
Final cleaning stage. Indexes all categorical features by integers.
Create embedding lookup tables.
'''

import json

import duckdb
import numpy as np
import pandas as pd

from hm_clothes_rec.paths import CLEANED_DIR, IMG_DIR, INDEXED_DIR

idx_dir = INDEXED_DIR

idx_dir.mkdir(parents=True, exist_ok=True) # make new indexed directory

customers = pd.read_parquet(str(CLEANED_DIR / 'customers_clean.parquet'))
articles = pd.read_parquet(str(CLEANED_DIR / 'articles_clean.parquet'))


# INDEXING ARTICLES DATA

item_cols = articles.columns.drop(['article_id', 'detail_desc'])

UNKNOWN = 'Unknown' 

item_emb_lookup = {} # {col: {value: idx}}, index 0 reserved for UNKNOWN in every col

# keep product codes / names shared by >= 3 articles, merge the rest into UNKNOWN (idx 0)
# MANUALLY CHECKED 
for col in ['product_code', 'prod_name']:
    col_counts = articles[col].value_counts()
    keep = col_counts[col_counts >= 3].index
    articles.loc[~articles[col].isin(keep), col] = UNKNOWN

# run thru every relevant column, find their uniques, and map the data to an integer index based on the unique values.
# MANUALLY CHECKED
for col in item_cols:
    # merge 'Unknown' / 'Undefined' / 'undefined' etc. into a single UNKNOWN label
    is_unknown = articles[col].str.lower().isin(['unknown', 'undefined'])
    articles.loc[is_unknown, col] = UNKNOWN

    uniques = [UNKNOWN] + sorted(set(articles[col].unique()) - {UNKNOWN})
    item_emb_lookup[col] = {value: i for i, value in enumerate(uniques)}
    articles[col] = articles[col].map(item_emb_lookup[col]).astype('int64')

# item stats from transactions (item_log_prob, median_price) are NOT computed here: they would see every split.
# splits.add_item_stats adds them afterwards from the training split only, in place in data/indexed (run splits.py after this)

# item_id = row position, the integer every model / embedding table is indexed by
# MANUALLY CHECKED
articles.insert(0, 'item_id', np.arange(len(articles), dtype='int64'))

# has_image = 1 if the article's image exists at images/{first 3 chars}/{article_id}.webp, else 0
# MANUALLY CHECKED 
img_dir = IMG_DIR
articles['has_image'] = [int((img_dir / aid[:3] / f'{aid}.webp').exists()) for aid in articles['article_id']]

# save indexed data + lookup tables (lookup sizes = embedding num_embeddings)
# MANUALLY CHECKED 
articles.to_parquet(idx_dir / 'articles_indexed.parquet', index=False)
articles[['item_id', 'article_id']].to_parquet(idx_dir / 'item_ids.parquet', index=False) # item_id <-> article_id
with open(idx_dir / 'item_emb_lookup.json', 'w') as f:
    json.dump(item_emb_lookup, f)

# INDEXING CUSTOMERS DATA
user_cols = customers.columns.drop(['customer_id'])

# merge 'None' into 'NONE' for news frequency
customers['fashion_news_frequency'] = customers['fashion_news_frequency'].replace('None', 'NONE')

# keep postal codes with >= 11 customers (~11k indices), merge the rest into UNKNOWN (idx 0), also serves as the bucket for unseen codes
# MANUALLY CHECKED 
postal_counts = customers['postal_code'].value_counts()
top_postal = postal_counts[postal_counts >= 11].index
customers.loc[~customers['postal_code'].isin(top_postal), 'postal_code'] = UNKNOWN

user_emb_lookup = {} # {col: {value: idx}}, index 0 reserved for UNKNOWN in every col

# MANUALLY CHECKED
for col in user_cols:
    # merge 'unknown' / 'Undefined' etc. into a single UNKNOWN label
    is_unknown = customers[col].str.lower().isin(['unknown', 'undefined'])
    customers.loc[is_unknown, col] = UNKNOWN

    uniques = [UNKNOWN] + sorted(set(customers[col].unique()) - {UNKNOWN})
    user_emb_lookup[col] = {value: i for i, value in enumerate(uniques)}
    customers[col] = customers[col].map(user_emb_lookup[col]).astype('int64')

# MANUALLY CHECKED
# user_id = row position, the integer the user tower is indexed by
customers.insert(0, 'user_id', np.arange(len(customers), dtype='int64'))

# MANUALLY CHECKED
# save indexed data + lookup tables
customers.to_parquet(idx_dir / 'customers_indexed.parquet', index=False)
customers[['user_id', 'customer_id']].to_parquet(idx_dir / 'user_ids.parquet', index=False) # user_id <-> customer_id
with open(idx_dir / 'user_emb_lookup.json', 'w') as f:
    json.dump(user_emb_lookup, f)

# MANUALLY CHECKED
# INDEXING TRANSACTIONS DATA
# swap the string ids for the integer ids above (done in duckdb, transactions is too big for pandas)
item_map = articles[['item_id', 'article_id']]
user_map = customers[['user_id', 'customer_id']]
tx_path = CLEANED_DIR / 'transactions_clean.parquet'
tx_idx_path = idx_dir / 'transactions_indexed.parquet'

duckdb.sql(f'''
    COPY (
        SELECT u.user_id, i.item_id, t.t_dat, t.price, t.sales_channel_id
        FROM '{tx_path}' t
        JOIN user_map u USING (customer_id)
        JOIN item_map i USING (article_id)
        ORDER BY t.t_dat
    ) TO '{tx_idx_path}' (FORMAT parquet)
''')

# MANUALLY CHECKED
# inner joins drop rows with ids missing from the lookup tables, there should be none
n_raw = duckdb.sql(f"SELECT COUNT(*) FROM '{tx_path}'").fetchall()[0][0]
n_idx = duckdb.sql(f"SELECT COUNT(*) FROM '{tx_idx_path}'").fetchall()[0][0]
assert n_raw == n_idx, f'{n_raw - n_idx} transactions have an article_id / customer_id not in the lookup tables'


