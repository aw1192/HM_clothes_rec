import kagglehub

from hm_clothes_rec.paths import RAW_DIR

# RUN ONLY ONCE

# Download latest version into <repo root>/data/archive/raw
RAW_DIR.mkdir(parents=True, exist_ok=True) # creates data/archive/ on a fresh clone
path = kagglehub.competition_download(
    'h-and-m-personalized-fashion-recommendations',
    output_dir=str(RAW_DIR),
)

print("Path to competition files:", path)