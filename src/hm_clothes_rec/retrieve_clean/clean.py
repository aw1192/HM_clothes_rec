import shutil
from pathlib import Path

import duckdb

from hm_clothes_rec.paths import CLEANED_DIR, DOWNSIZED_DIR, IMG_DIR

ITEMS = DOWNSIZED_DIR / 'articles.parquet'
USERS = DOWNSIZED_DIR / 'customers.parquet'
INTERACTIONS = DOWNSIZED_DIR / 'transactions_train.parquet'
IMAGES = DOWNSIZED_DIR / 'images'

OUT_DIR = CLEANED_DIR


def clean_customers(src=USERS, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = out_dir / 'customers_clean.parquet'

    # drop the FN and Active
    # label missing categories as unknown
    # then bucket the ages
    duckdb.read_parquet(str(src)).select('''
        * EXCLUDE (FN, Active, age)
          REPLACE (
            COALESCE(club_member_status, 'unknown') AS club_member_status,
            COALESCE(fashion_news_frequency, 'unknown') AS fashion_news_frequency
          ),
        CASE
            WHEN age IS NULL THEN 'unknown'
            WHEN age < 20 THEN '16-19'
            WHEN age < 25 THEN '20-24'
            WHEN age < 30 THEN '25-29'
            WHEN age < 40 THEN '30-39'
            WHEN age < 50 THEN '40-49'
            WHEN age < 60 THEN '50-59'
            ELSE '60+'
        END AS age_bucket
    ''').write_parquet(str(dest)) # write to the destination 
    return dest

def clean_articles(src=ITEMS, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = out_dir / 'articles_clean.parquet'

    duckdb.read_parquet(str(src)).select('''
                * EXCLUDE (product_type_no, graphical_appearance_no, 
                colour_group_code, perceived_colour_value_id, perceived_colour_master_id,
                department_no, index_code, index_group_no, section_no, garment_group_no)
                REPLACE (COALESCE(detail_desc, 'unknown') AS detail_desc)
''').write_parquet(str(dest))
    return dest

def clean_transactions(src=INTERACTIONS, out_dir=OUT_DIR):
    '''
    transactions need no cleaning (no missing values), copied as is so every downstream step reads cleaned/
    '''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = out_dir / 'transactions_clean.parquet'
    shutil.copyfile(src, dest)
    return dest

def copy_images(src=IMAGES, dest=IMG_DIR):
    '''
    the downsized .webp thumbnails, same {first 3 chars}/{article_id}.webp layout. files already there are kept
    '''
    shutil.copytree(src, dest, dirs_exist_ok=True)
    return dest

if __name__ == "__main__":
    print(clean_customers())
    print(clean_articles())
    print(clean_transactions())
    print(copy_images())


