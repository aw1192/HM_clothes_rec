'''
Every data directory in one place, import from here instead of building paths per script.

data/
  archive/raw/        kaggle download (download_and_store.py)
  archive/downsized/  csv -> parquet, shrunk images (downsize_data.py)
  cleaned/            cleaned tables + images (clean.py)
  indexed/            integer ids, lookup tables, item stats from train_retrieval, CLIP vectors.
                      the single copy every model reads (index_data.py, then splits.py, image_text_emb.py)
  splits/             train_retrieval / train_rank / test transaction parquets (splits.py)
'''

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] # repo root
DATA_DIR = ROOT / 'data'

ARCHIVE_DIR = DATA_DIR / 'archive'
RAW_DIR = ARCHIVE_DIR / 'raw'
DOWNSIZED_DIR = ARCHIVE_DIR / 'downsized'

CLEANED_DIR = DATA_DIR / 'cleaned'
IMG_DIR = CLEANED_DIR / 'images'

INDEXED_DIR = DATA_DIR / 'indexed'
SPLITS_DIR = DATA_DIR / 'splits'
