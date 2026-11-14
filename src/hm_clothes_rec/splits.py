from datetime import timedelta
from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parents[2] / 'data' / 'cleaned'

ITEMS = DATA_DIR / 'articles_clean.parquet'
USERS = DATA_DIR / 'customers_clean.parquet'
INTERACTIONS = DATA_DIR / 'transactions_clean.parquet'

OUT_DIR = DATA_DIR.parent / 'splits'


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
