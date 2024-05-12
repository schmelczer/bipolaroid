from pathlib import Path
from typing import List, Tuple
from torch.utils.data import DataLoader, random_split
from editor.training import HistogramDataset
import logging
import torch
from config import CACHE_PATH
import os


def create_data_loaders(
    data: List[Path],
    edit_count: int,
    bin_count: int,
    training_batch_size: int,
    train_size=0.9,
    delete_corrupt_images: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    dataset = HistogramDataset(
        data,
        edit_count=edit_count,
        bin_count=bin_count,
        delete_corrupt_images=delete_corrupt_images,
        cache_path=CACHE_PATH,
    )
    total_size = len(dataset)
    train_size = int(train_size * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count()
    )
    logging.info(
        f"Loaded {len(train_dataset)} training images and {len(test_dataset)} test images"
    )

    return train_data_loader, test_data_loader
