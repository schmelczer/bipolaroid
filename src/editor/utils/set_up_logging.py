import logging
from datetime import datetime
from typing import Optional


def set_up_logging(logs_path: Optional[str] = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            (
                logging.FileHandler(
                    logs_path / f"{datetime.now().isoformat(timespec='minutes')}.log"
                )
                if logs_path
                else logging.NullHandler()
            ),
        ],
    )
