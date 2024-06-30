from torch.utils.data import Dataset
from typing import List, Optional, Tuple
from utils import compute_histogram
from operations.random_edit import random_edit
from PIL import Image
from tqdm import tqdm
import logging
import torch
from pathlib import Path

import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None


class HistogramDataset(Dataset):
    def __init__(
        self,
        /,
        paths: List[Path],
        bin_count: int = 16,
        edit_count: int = 12,
        target_size=(240, 240),
        delete_corrupt_images: bool = False,
        cache_path: Optional[Path] = None,
    ):
        self._paths = sorted(paths)
        logging.info(f"Loaded {len(self._paths)} original images")

        self._edit_count = edit_count
        self._bin_count = bin_count
        self._target_size = target_size
        self._cache_path = cache_path
        if self._cache_path:
            self._cache_path = (
                self._cache_path
                / f"{self._bin_count}bins_{self._target_size[0]}x{self._target_size[1]}px"
            )

        if delete_corrupt_images:
            self._delete_corrupt_images()

    def _delete_corrupt_images(self) -> None:
        deleted_count = 0
        for path in tqdm(self._paths):
            try:
                Image.open(path)
            except:
                logging.warning(f"Failed to open {path}, deleting...")
                deleted_count += 1
                path.unlink()
        logging.info(f"Deleted {deleted_count} corrupt images")

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
        return random_edit(original_image, seed=original_idx * 7919 + edit_idx)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        original_idx = idx // self._edit_count
        edit_idx = idx % self._edit_count

        if self._cache_path is not None:
            _cached_data_path = self._cache_path / str(original_idx) / f"{edit_idx}.pt"
            _cached_data_path.parent.mkdir(parents=True, exist_ok=True)
            if _cached_data_path.exists():
                try:
                    return torch.load(_cached_data_path)
                except:
                    logging.warning(
                        f"Failed to load {_cached_data_path}, regenerating..."
                    )

        edited = self.get_edited_image(original_idx, edit_idx)
        edited_histogram = compute_histogram(
            edited, bins=self._bin_count, normalize=True
        )

        original = self.get_original_image(original_idx)
        original_histogram = compute_histogram(
            original, bins=self._bin_count, normalize=True
        )

        result = (
            torch.tensor(edited_histogram, dtype=torch.float).unsqueeze(0),
            torch.tensor(original_histogram, dtype=torch.float).unsqueeze(0),
        )

        if self._cache_path is not None:
            torch.save(result, _cached_data_path)

        return result
