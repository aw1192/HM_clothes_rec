'''
Final cleaning stage. Indexes all categorical features by integers.
Create embedding lookup tables.
'''

# TODO: RERUN THIS FILE AFTER DATA SPLITS ARE CREATED

import json

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / 'data'

idx_dir = DATA_DIR / 'indexed'

idx_dir.mkdir(parents=True, exist_ok=True) # make new indexed directory

customers = pd.read_parquet(str(DATA_DIR / 'cleaned' / 'customers_clean.parquet'))
articles = pd.read_parquet(str(DATA_DIR / 'cleaned' / 'articles_clean.parquet'))


# INDEXING ARTICLES DATA

item_cols = articles.columns.drop(['article_id', 'detail_desc'])

UNKNOWN = 'Unknown' 

item_emb_lookup = {} # {col: {value: idx}}, index 0 reserved for UNKNOWN in every col

# keep product codes / names shared by >= 3 articles, merge the rest into UNKNOWN (idx 0)
for col in ['product_code', 'prod_name']:
    col_counts = articles[col].value_counts()
    keep = col_counts[col_counts >= 3].index
    articles.loc[~articles[col].isin(keep), col] = UNKNOWN

# run thru every relevant column, find their uniques, and map the data to an integer index based on the unique values.
for col in item_cols:
    # merge 'Unknown' / 'Undefined' / 'undefined' etc. into a single UNKNOWN label
    is_unknown = articles[col].str.lower().isin(['unknown', 'undefined'])
    articles.loc[is_unknown, col] = UNKNOWN

    uniques = [UNKNOWN] + sorted(set(articles[col].unique()) - {UNKNOWN})
    item_emb_lookup[col] = {value: i for i, value in enumerate(uniques)}
    articles[col] = articles[col].map(item_emb_lookup[col]).astype('int64')

# item probability = (# transactions with item) / (total # transactions)
# median price = median of the item's transaction prices
item_counts = duckdb.sql(f'''
    SELECT article_id, COUNT(*) AS cnt, MEDIAN(price) AS median_price
    FROM '{DATA_DIR / 'cleaned' / 'transactions_clean.parquet'}'
    GROUP BY article_id
''').df()

# add item probability and median price into the articles table
articles = articles.merge(item_counts, on='article_id', how='left')
item_prob = (articles['cnt'].fillna(0) + 1) / (item_counts['cnt'].sum() + len(articles)) # add-one smoothing
articles['item_log_prob'] = np.log(item_prob) # log q for sampling bias correction
articles = articles.drop(columns='cnt')

# items with no transactions get the median price of their product type
type_median = articles.groupby('product_type_name')['median_price'].transform('median')
articles['median_price'] = articles['median_price'].fillna(type_median).fillna(articles['median_price'].median())

# save indexed data + lookup tables (lookup sizes = embedding num_embeddings)
articles.to_parquet(idx_dir / 'articles_indexed.parquet', index=False)
with open(idx_dir / 'item_emb_lookup.json', 'w') as f:
    json.dump(item_emb_lookup, f)

# INDEXING CUSTOMERS DATA
user_cols = customers.columns.drop(['customer_id'])

# merge 'None' into 'NONE' for news frequency
customers['fashion_news_frequency'] = customers['fashion_news_frequency'].replace('None', 'NONE')

# keep postal codes with >= 11 customers (~11k indices), merge the rest into UNKNOWN (idx 0), also serves as the bucket for unseen codes
postal_counts = customers['postal_code'].value_counts()
top_postal = postal_counts[postal_counts >= 11].index
customers.loc[~customers['postal_code'].isin(top_postal), 'postal_code'] = UNKNOWN

user_emb_lookup = {} # {col: {value: idx}}, index 0 reserved for UNKNOWN in every col

for col in user_cols:
    # merge 'unknown' / 'Undefined' etc. into a single UNKNOWN label
    is_unknown = customers[col].str.lower().isin(['unknown', 'undefined'])
    customers.loc[is_unknown, col] = UNKNOWN

    uniques = [UNKNOWN] + sorted(set(customers[col].unique()) - {UNKNOWN})
    user_emb_lookup[col] = {value: i for i, value in enumerate(uniques)}
    customers[col] = customers[col].map(user_emb_lookup[col]).astype('int64')

# save indexed data + lookup tables
customers.to_parquet(idx_dir / 'customers_indexed.parquet', index=False)
with open(idx_dir / 'user_emb_lookup.json', 'w') as f:
    json.dump(user_emb_lookup, f)


