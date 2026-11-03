import time
from pathlib import Path

import duckdb
from PIL import Image, ImageOps
from tqdm import tqdm

ID_COLS = {'article_id': 'VARCHAR', 'product_code': 'VARCHAR'}

# <repo root>/data
DATA_DIR = Path(__file__).resolve().parents[2] / 'data' # get the data directory 
RAW_DIR = DATA_DIR / 'raw'
OUT_DIR = DATA_DIR / 'downsized'

def csv_to_parquet(csv_path, out_path):
    csv_path = Path(csv_path)
    out_path = Path(out_path)

    # make directories
    out_path.mkdir(parents=True, exist_ok=True)

    # destination path
    dest = out_path / f'{csv_path.stem}.parquet'

    # save csv as a parquet
    cols = duckdb.read_csv(str(csv_path)).columns
    dtype = {c: t for c, t in ID_COLS.items() if c in cols}

    duckdb.read_csv(str(csv_path), dtype=dtype).write_parquet(str(dest))
    return dest


def downsize_img(src_path, dest_path):
    src, dst = Path(src_path), Path(dest_path)

    for p in tqdm(list(src.rglob('*.jpg'))): # searching the source folder for jpgs
        out = dst / p.relative_to(src).with_suffix('.webp') # create the output destination

        if out.exists(): # skip if the image is there already
            continue    

        out.parent.mkdir(parents=True, exist_ok=True) # create the folder for thumbnail, and its parents

        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert('RGB') # convert to RGB in case, and transpose the image if it is sideways
            im.thumbnail((400, 600)) # shrink image
            im.save(out, 'WEBP', quality = 80) # save thumbnail

    return dst

if __name__ == "__main__":
    s = time.time()

    articles = csv_to_parquet(RAW_DIR / 'articles.csv', OUT_DIR)
    customers = csv_to_parquet(RAW_DIR / 'customers.csv', OUT_DIR)
    transactions = csv_to_parquet(RAW_DIR / 'transactions_train.csv', OUT_DIR)
    images = downsize_img(RAW_DIR / 'images', OUT_DIR / 'images')

    e = time.time()

    print(articles, customers, transactions, images, e-s)