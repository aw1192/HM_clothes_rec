'''
Runs the data pipeline in order. Every step creates the folders it writes to.

  download   kaggle csvs + images          -> data/archive/raw
  downsize   csv -> parquet, small images  -> data/archive/downsized
  clean      cleaned tables + images       -> data/cleaned
  index      integer ids + lookup tables   -> data/indexed
  splits     time splits + item stats      -> data/splits, data/indexed/articles_indexed.parquet
  embed      FashionCLIP item vectors      -> data/indexed/clip_*.npy   (use a gpu)

usage:
  uv run python -m hm_clothes_rec.pipeline                  # everything except download
  uv run python -m hm_clothes_rec.pipeline --from index     # rerun from a step onwards
  uv run python -m hm_clothes_rec.pipeline --only splits    # one step
  uv run python -m hm_clothes_rec.pipeline --download       # include the kaggle download (run once)
'''

import argparse
import runpy
import time

# step name -> module, run as if by `python -m <module>`
STEPS = {
    'download': 'hm_clothes_rec.retrieve_clean.download_and_store',
    'downsize': 'hm_clothes_rec.retrieve_clean.downsize_data',
    'clean': 'hm_clothes_rec.retrieve_clean.clean',
    'index': 'hm_clothes_rec.retrieve_clean.index_data',
    'splits': 'hm_clothes_rec.retrieve_clean.splits',
    'embed': 'hm_clothes_rec.image_text_emb',
}


def run(steps):
    for name in steps:
        print(f'\n===== {name} ({STEPS[name]}) =====')
        start = time.time()
        runpy.run_module(STEPS[name], run_name='__main__', alter_sys=True)
        print(f'===== {name} done in {time.time() - start:.0f}s =====')


def main():
    parser = argparse.ArgumentParser(description='Run the H&M data pipeline.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--from', dest='start', choices=STEPS, help='run this step and every step after it')
    group.add_argument('--only', choices=STEPS, help='run just this step')
    parser.add_argument('--download', action='store_true', help='include the kaggle download (skipped by default)')
    args = parser.parse_args()

    names = list(STEPS)
    if args.only:
        steps = [args.only]
    else:
        steps = names[names.index(args.start):] if args.start else names
        if not args.download and args.start != 'download':
            steps = [s for s in steps if s != 'download']
    run(steps)


if __name__ == '__main__':
    main()
