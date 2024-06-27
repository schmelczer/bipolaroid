from datetime import timedelta
import logging
from pathlib import Path
from random import choice
from itertools import count
import json
from typing import Any, Dict, List
from .train import train
from .get_next_run_name import get_next_run_name
from models import save_model
from .get_data_loader import get_data_loader
import torch


def random_hparam_search(
    hyperparameters: List[Dict[str, Any]],
    train_data_paths: List[Path],
    test_data_paths: List[Path],
    models_path: Path,
    tensorboard_path: Path,
    timeout_hours: int,
    device: torch.device,
) -> None:
    for _ in count():
        current_hyperparameters = {
            k: v.rvs() if hasattr(v, "rvs") else choice(v)
            for k, v in choice(hyperparameters).items()
        }
        serialized_hparams = json.dumps(
            current_hyperparameters, indent=2, sort_keys=True
        )
        logging.info(
            f"Starting {get_next_run_name(tensorboard_path)} with hparams {serialized_hparams}"
        )

        log_dir = tensorboard_path / get_next_run_name(tensorboard_path)

        try:
            model = train(
                hyperparameters=current_hyperparameters,
                train_data_paths=train_data_paths,
                test_data_paths=test_data_paths,
                max_duration=timedelta(hours=timeout_hours),
                log_dir=log_dir,
                use_tqdm=False,
                device=device,
                **current_hyperparameters,
            )
            model_path = models_path / get_next_run_name(models_path)
            save_model(model, current_hyperparameters, model_path)
            del model
        except KeyboardInterrupt as e:
            logging.info("Interrupted, stopping")
            break
        except TimeoutError as e:
            logging.warning(f"Timeout, aborting experiment")
        except Exception as e:
            logging.error(
                f"Error with hparams {current_hyperparameters}:\n\t{e}", stack_info=True
            )
