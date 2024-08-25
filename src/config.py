import random
from pathlib import Path

# DATA = sorted(Path("/mnt/wsl/PHYSICALDRIVE0p1/downloaded-unsplash").glob("*"))
DATA = sorted(Path("/mnt/wsl/PHYSICALDRIVE0p1/featured").glob("*"))

TRAIN_SIZE = 0.9

CACHE_PATH = Path("/mnt/wsl/PHYSICALDRIVE1/data/cache2")
MODELS_PATH = Path("/home/andras/projects/bipolaroid/saved_models")
LOGS_PATH = Path("/home/andras/projects/bipolaroid/logs")
RUNS_PATH = Path("/home/andras/projects/bipolaroid/runs")


for path in [CACHE_PATH, MODELS_PATH, LOGS_PATH, RUNS_PATH]:
    path.mkdir(exist_ok=True, parents=True)


length = len(DATA)
indices = list(range(length))

random.seed(42)
random.shuffle(indices)

train_indices = indices[: int(length * TRAIN_SIZE)]
test_indices = indices[int(length * TRAIN_SIZE) :]

TRAIN_DATA = [DATA[i] for i in train_indices]
TEST_DATA = [DATA[i] for i in test_indices]
