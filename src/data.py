import random
from config import DATA, TRAIN_SIZE

random.seed(42)
length = len(DATA)
indices = list(range(length))

random.shuffle(indices)

train_indices = indices[: int(length * TRAIN_SIZE)]
test_indices = indices[int(length * TRAIN_SIZE) :]

TRAIN_DATA = [DATA[i] for i in train_indices]
TEST_DATA = [DATA[i] for i in test_indices]
