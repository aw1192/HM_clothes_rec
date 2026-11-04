from pathlib import Path

import kagglehub

# RUN ONLY ONCE 

# <repo root>/data/raw
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Download latest version
path = kagglehub.competition_download(
    'h-and-m-personalized-fashion-recommendations',
    output_dir=str(DATA_DIR),
)

print("Path to competition files:", path)