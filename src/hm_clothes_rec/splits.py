from datetime import date
from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parents[2] / 'data' / 'cleaned'

ITEMS = DATA_DIR / 'articles_clean.parquet'
USERS = DATA_DIR / 'customers_clean.parquet'
INTERACTIONS = DATA_DIR / 'transactions_clean.parquet'

# name -> (first day, last day), both inclusive
SPLITS = {
    'train_retrieval': (date(2018, 9, 20), date(2020, 9, 8)),
    'train_ranker': (date(2020, 9, 9), date(2020, 9, 15)),
    'validation': (date(2020, 9, 16), date(2020, 9, 22)),
}

# the last day of data a model may see when it predicts each window
HISTORY_END = {
    'train_ranker': date(2020, 9, 8),
    'validation': date(2020, 9, 15),
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
    Every row a model may use to predict the target split.

    Popularity and repurchase use this rather than train_retrieval, because
    predicting validation is allowed to use the ranker week.
    '''
    return _between(SPLITS['train_retrieval'][0], HISTORY_END[target])


if __name__ == "__main__":
    total = 0
    for name, (lo, hi) in SPLITS.items():
        n = load_split(name).count('*').fetchall()[0][0]
        total += n
        print(f'{name:16} {lo} to {hi} {n:>12,}')

    # the splits must cover every row exactly once
    all_rows = duckdb.read_parquet(str(INTERACTIONS)).count('*').fetchall()[0][0]
    assert total == all_rows, f'splits cover {total:,} of {all_rows:,} rows'
    print(f'{"total":16} {"":24} {total:>12,}')
