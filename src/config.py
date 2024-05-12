from pathlib import Path

DATA = sorted(Path("/mnt/wsl/PHYSICALDRIVE1/data/unsplash").glob("*.jpg"))

CACHE_PATH = Path("/mnt/wsl/PHYSICALDRIVE1/data/cache2")
CACHE_PATH.mkdir(exist_ok=True, parents=True)

MODELS_PATH = Path("/home/andras/projects/bipolaroid/models")
MODELS_PATH.mkdir(exist_ok=True, parents=True)
