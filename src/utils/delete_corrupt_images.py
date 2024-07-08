import logging
from pathlib import Path
from typing import List
from PIL import Image
import PIL.Image
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = None


def delete_corrupt_images(paths: List[Path]) -> None:
    deleted_count = 0
    for path in tqdm(paths):
        if not path.exists():
            logging.warning(f"{path} does not exist, skipping...")
            continue
        try:
            Image.open(path)
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt, exiting...")
            raise
        except:
            logging.warning(f"Failed to open {path}, deleting...")
            deleted_count += 1
            path.unlink()
    logging.info(f"Deleted {deleted_count} corrupt images")
