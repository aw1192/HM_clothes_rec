from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from hm_clothes_rec.paths import INDEXED_DIR, SPLITS_DIR

INTERACTIONS = INDEXED_DIR / 'transactions_indexed.parquet'
ARTICLES = INDEXED_DIR / 'articles_indexed.parquet'

OUT_DIR = SPLITS_DIR

# split the item stats (log q, median price) are computed from. only this split's rows are seen
STATS_SPLIT = 'train_retrieval'


def _date_range():
    return duckdb.read_parquet(str(INTERACTIONS)).aggregate(
        'min(t_dat), max(t_dat)'
    ).fetchall()[0]


FIRST_DAY, LAST_DAY = _date_range()  # 2018-09-20, 2020-09-22

# test: the most recent 7 days in the data
TEST_START = LAST_DAY - timedelta(days=6)
# train_rank: the 7 days before test
RANK_START = TEST_START - timedelta(days=7)

# name -> (first day, last day), both inclusive
SPLITS = {
    'train_retrieval': (FIRST_DAY, RANK_START - timedelta(days=1)),
    'train_rank': (RANK_START, TEST_START - timedelta(days=1)),
    'test': (TEST_START, LAST_DAY),
}


def _between(lo, hi):
    return duckdb.read_parquet(str(INTERACTIONS)).filter(
        f"t_dat BETWEEN DATE '{lo}' AND DATE '{hi}'"
    )


def load_split(name):
    '''Rows of one split, as a lazy duckdb relation.'''
    lo, hi = SPLITS[name]
    return _between(lo, hi)


def load_history(target):
    '''
    Every row a model may use to predict the target split: everything before it.

    Popularity and repurchase use this rather than train_retrieval, because
    predicting test is allowed to use the train_rank week.
    '''
    return _between(FIRST_DAY, SPLITS[target][0] - timedelta(days=1))


def save_splits(out_dir=OUT_DIR):
    '''Write each split to its own parquet file, e.g. data/splits/test.parquet.'''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name in SPLITS:
        dest = out_dir / f'{name}.parquet'
        load_split(name).write_parquet(str(dest))
        paths[name] = dest
    return paths


def add_item_stats(path=ARTICLES, stats_split=STATS_SPLIT):
    '''
    Add the item stats the models use to data/indexed/articles_indexed.parquet, in place,
    computed from the stats_split rows only so no later split leaks in:
      item_log_prob: log of (# transactions with item + 1) / (# transactions + # items), add-one smoothed
      median_price:  median of the item's transaction prices. items never bought in stats_split get the
                     median of their product type, or if still NA, the median over all items
    safe to rerun: existing stat columns are dropped and rebuilt
    '''
    path = Path(path)

    # errors='ignore' so a fresh index_data.py output (no stat cols yet) and a rerun both work
    articles = pd.read_parquet(path).drop(
        columns=['median_price', 'item_log_prob'], errors='ignore'
    )

    item_counts = load_split(stats_split).aggregate(
        'item_id, COUNT(*) AS cnt, MEDIAN(price) AS median_price', 'item_id'
    ).df()

    articles = articles.merge(item_counts, on='item_id', how='left') # left join keeps row order = item_id
    item_prob = (articles['cnt'].fillna(0) + 1) / (item_counts['cnt'].sum() + len(articles))
    articles['item_log_prob'] = np.log(item_prob) # log q for sampling bias correction
    articles = articles.drop(columns='cnt')

    type_median = articles.groupby('product_type_name')['median_price'].transform('median')
    articles['median_price'] = articles['median_price'].fillna(type_median).fillna(articles['median_price'].median())

    assert (articles['item_id'].to_numpy() == np.arange(len(articles))).all(), 'row position must equal item_id'
    articles.to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    paths = save_splits()

    total = 0
    for name, (lo, hi) in SPLITS.items():
        n = duckdb.read_parquet(str(paths[name])).count('*').fetchall()[0][0]
        total += n
        print(f'{name:16} {lo} to {hi} {n:>12,}  -> {paths[name]}')

    # the splits must cover every row exactly once
    all_rows = duckdb.read_parquet(str(INTERACTIONS)).count('*').fetchall()[0][0]
    assert total == all_rows, f'splits cover {total:,} of {all_rows:,} rows'
    print(f'{"total":16} {"":24} {total:>12,}')

    articles_path = add_item_stats()
    print(f'{STATS_SPLIT} item stats -> {articles_path}')

