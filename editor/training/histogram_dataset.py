from torch.utils.data import Dataset
from typing import List, Optional, Tuple
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
        cache_path: Optional[Path] = None,
    ):
        self._paths = sorted(paths)
        self._edit_count = edit_count
        self._bin_count = bin_count
        self._target_size = target_size
        self._cache_path = cache_path

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

    def get_original_image(self, original_idx: int) -> Image.Image:
        original_path = self._paths[original_idx]
        original = Image.open(original_path)
        original.thumbnail(
            self._target_size, Image.Resampling.LANCZOS
        )  # size will be at most target_size, the aspect ratio is preserved
        return original

    def get_edited_image(self, original_idx: int, edit_idx: int) -> Image.Image:
        original_image = self.get_original_image(original_idx)
        return random_edit(original_image, seed=edit_idx)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache_path is not None:
            self._cached_data_path = self._cache_path / f"{idx}.pt"
            if self._cached_data_path.exists():
                try:
                    return torch.load(self._cached_data_path)
                except:
                    print(f"Failed to load {self._cached_data_path}, regenerating...")

        original_idx = idx // self._edit_count
        original = self.get_original_image(original_idx)
        edited = random_edit(original, seed=idx)

        edited_histogram = compute_histogram(
            edited, bins=self._bin_count, normalize=True
        )

        original_histogram = compute_histogram(
            original, bins=self._bin_count, normalize=True
        )

        result = (
            torch.tensor(edited_histogram, dtype=torch.float).unsqueeze(0),
            torch.tensor(original_histogram, dtype=torch.float).unsqueeze(0),
        )

        if self._cache_path is not None:
            torch.save(result, self._cached_data_path)

        return result
