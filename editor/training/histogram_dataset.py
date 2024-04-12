from torch.utils.data import Dataset
from typing import List
from editor.utils import compute_histogram
from .random_edit import random_edit
from PIL import Image
from tqdm import tqdm
import torch
from pathlib import Path

import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None


class HistogramDataset(Dataset):
    def __init__(
        self,
        paths: List[Path],
        edit_count: int = 5,
        bin_count: int = 32,
        target_size=(480, 480),
        delete_corrupt_images: bool = False,
    ):
        self._paths = sorted(paths)
        self._edit_count = edit_count
        self._bin_count = bin_count
        self._target_size = target_size

        if delete_corrupt_images:
            self._delete_corrupt_images()

    def _delete_corrupt_images(self) -> None:
        deleted_count = 0
        for path in tqdm(self._paths):
            try:
                Image.open(path)
            except:
                print(f"Failed to open {path}, deleting...")
                deleted_count += 1
                path.unlink()
        print(f"Deleted {deleted_count} corrupt images")

    def __len__(self):
        return len(self._paths) * self._edit_count

    def __getitem__(self, idx):
        original_idx = idx // self._edit_count
        original_path = self._paths[original_idx]
        original = Image.open(original_path)
        original.thumbnail(self._target_size, Image.Resampling.LANCZOS)

        edited = random_edit(original, seed=idx)

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
