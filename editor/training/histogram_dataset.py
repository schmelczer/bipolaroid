from torch.utils.data import Dataset
from typing import Generator, Tuple, List
from editor.utils import compute_histogram
from PIL import Image
from tqdm import tqdm
import torch
from pathlib import Path


class HistogramDataset(Dataset):
    def __init__(
        self, paths: List[Path], expected_edit_count: int = 5, bin_count: int = 32
    ):
        self._paths = paths
        self._expected_edit_count = expected_edit_count
        self._bin_count = bin_count
        self._pairs = list(self._get_pairs())

    def _get_pairs(self) -> Generator[Tuple[Path, Path], None, None]:
        for path in tqdm(self._paths):
            if len(list(path.glob("*.jpg"))) != self._expected_edit_count + 1:
                continue

            original_path = path / "original.jpg"
            try:
                Image.open(original_path)
            except:
                print(f"Failed to open {original_path}")
                continue
            yield original_path, original_path  # The model should leave the original image unchanged
            for i in range(self._expected_edit_count):
                try:
                    Image.open(path / f"{i}.jpg")
                except:
                    print(f'Failed to open {path / f"{i}.jpg"}')
                    break
                yield original_path, path / f"{i}.jpg"

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        original, edited = self._pairs[idx]
        original_histogram = compute_histogram(
            original, bins=self._bin_count, normalize=True
        )
        edited_histogram = compute_histogram(
            edited, bins=self._bin_count, normalize=True
        )
        return (
            torch.tensor(edited_histogram, dtype=torch.float).unsqueeze(0),
            torch.tensor(original_histogram, dtype=torch.float).unsqueeze(0),
        )
