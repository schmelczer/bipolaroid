from pathlib import Path
from typing import List
from torch.utils.data import DataLoader
from config import CACHE_PATH
from training import HistogramDataset
import os


def get_data_loader(
    data: List[Path], edit_count: int, bin_count: int, batch_size: int, **_
) -> DataLoader:
    return DataLoader(
        dataset=HistogramDataset(
            paths=data,
            edit_count=edit_count,
            bin_count=bin_count,
            cache_path=CACHE_PATH,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )
