from torch.utils.data import Dataset
from typing import List, Optional, Tuple
from utils import compute_histogram
from operations.random_edit import random_edit
from PIL import Image
import logging
import torch
from pathlib import Path
import struct
import zlib
import PIL.Image
import numpy as np

PIL.Image.MAX_IMAGE_PIXELS = None


class HistogramDataset(Dataset):
    def __init__(
        self,
        /,
        paths: List[Path],
        bin_count: int = 16,
        edit_count: int = 12,
        target_size=(240, 240),
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

        cached_data_path = None
        if self._cache_path is not None:
            cached_data_path = self._cache_path / str(original_idx) / f"{edit_idx}.bin"
            cached_data_path.parent.mkdir(parents=True, exist_ok=True)

        if cached_data_path and cached_data_path.exists():
            try:
                edited_histogram, original_histogram = self.read_2_histograms(
                    cached_data_path, self._bin_count
                )
            except:
                logging.warning(f"Failed to load {cached_data_path}, regenerating...")
        else:
            edited = self.get_edited_image(original_idx, edit_idx)
            edited_histogram = compute_histogram(
                edited, bins=self._bin_count, normalize=True
            )

            original = self.get_original_image(original_idx)
            original_histogram = compute_histogram(
                original, bins=self._bin_count, normalize=True
            )

            if cached_data_path:
                self.save_2_histograms(
                    edited_histogram,
                    original_histogram,
                    cached_data_path,
                )

        return (
            torch.tensor(edited_histogram, dtype=torch.float).unsqueeze(0),
            torch.tensor(original_histogram, dtype=torch.float).unsqueeze(0),
        )

    @staticmethod
    def save_2_histograms(tensor1: np.ndarray, tensor2: np.ndarray, path: Path):
        flat_array1 = tensor1.flatten().astype(np.float32)
        flat_array2 = tensor2.flatten().astype(np.float32)

        assert len(flat_array1) == len(flat_array2)

        format = f"{len(flat_array1)}f{len(flat_array2)}f"
        packed_bytes = struct.pack(format, *flat_array1, *flat_array2)
        compressed_bytes = zlib.compress(packed_bytes, level=9)
        with open(path, "wb") as f:
            f.write(compressed_bytes)

    @staticmethod
    def read_2_histograms(path: Path, bin_count: int) -> Tuple[np.ndarray, np.ndarray]:
        length = bin_count**3
        format = f"{length}f{length}f"
        with open(path, "rb") as f:
            packed_data = f.read()

        unpacked_data = struct.unpack(format, zlib.decompress(packed_data))
        return (
            np.array(unpacked_data[:length], dtype=np.float32).reshape(
                (bin_count, bin_count, bin_count)
            ),
            np.array(unpacked_data[length:], dtype=np.float32).reshape(
                (bin_count, bin_count, bin_count)
            ),
        )
