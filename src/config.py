from pathlib import Path

DATA = sorted(Path("/mnt/wsl/PHYSICALDRIVE1/data/unsplash").glob("*.jpg"))
TRAIN_SIZE = 0.9

CACHE_PATH = Path("/mnt/wsl/PHYSICALDRIVE1/data/cache2")
MODELS_PATH = Path("/home/andras/projects/bipolaroid/models")
LOGS_PATH = Path("/home/andras/projects/bipolaroid/logs")
RUNS_PATH = Path("/home/andras/projects/bipolaroid/runs")


for path in [
    CACHE_PATH,
    MODELS_PATH,
    LOGS_PATH,
    RUNS_PATH
]:
    path.mkdir(exist_ok=True, parents=True)
