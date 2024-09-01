from typing import Any, Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.optim import Adam
from .get_next_run_name import get_next_run_name
from utils import serialise_hparams
from visualisation import plot_histograms_in_2d
from models import create_model, load_model, save_model
import torch
from .get_data_loader import get_data_loader
from ray import train
from ray.train import Checkpoint
import os
import tempfile
from more_itertools import distribute


EPSILON = 1e-5


def train_with_ray_factory(
    train_data_paths: List[Path],
    test_data_paths: List[Path],
    device: torch.device,
    log_dir: Path,
):
    def train_with_ray(hyperparameters: Dict[str, Any]):
        def inner(
            hyperparameters: Dict[str, Any],
            chunk_count: int,
            **_,
        ) -> torch.nn.Module:
            train_data_loader = get_data_loader(train_data_paths, **hyperparameters)
            test_data_loader = get_data_loader(
                test_data_paths, **{**hyperparameters, "edit_count": 1}
            )
            examples = next(iter(test_data_loader))

            model, optimizer, scheduler, start_chunk_id, run_name = (
                load_or_create_state(
                    device=device,
                    log_dir=log_dir,
                    **hyperparameters,
                )
            )
            loss_function = torch.nn.KLDivLoss(reduction="batchmean").to(device)

            with SummaryWriter(log_dir=log_dir / run_name) as writer:
                writer.add_graph(model, examples[0].to(device))
                for chunk_id, chunk in enumerate(
                    distribute(chunk_count, train_data_loader)[start_chunk_id:-1],
                    start=start_chunk_id,
                ):
                    chunk_training_loss = 0
                    writer.add_scalar(
                        "Actual learning rate",
                        scheduler.get_last_lr()[0],
                        chunk_id,
                    )
                    for batch_id, (edited_histogram, original_histogram) in enumerate(
                        chunk
                    ):
                        global_step = (
                            chunk_id * (len(train_data_loader) // chunk_count)
                            + batch_id
                        )
                        optimizer.zero_grad()
                        predicted_original = model(edited_histogram.to(device))
                        loss = loss_function(
                            torch.log(predicted_original + EPSILON),
                            original_histogram.to(device),
                        )

                        chunk_training_loss += loss.item()
                        writer.add_scalar(
                            "Loss/train/batch", loss, global_step=global_step
                        )
                        loss.backward()
                        optimizer.step()

                    with torch.no_grad():
                        model.eval()
                        write_histograms(
                            model=model,
                            examples=examples,
                            writer=writer,
                            device=device,
                            global_step=global_step,
                        )
                        chunk_test_loss = 0
                        for (
                            edited_histogram,
                            original_histogram,
                        ) in test_data_loader:
                            predicted_original = model(edited_histogram.to(device))
                            chunk_test_loss += loss_function(
                                torch.log(predicted_original + EPSILON),
                                original_histogram.to(device),
                            ).item()
                        model.train()

                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        temp_checkpoint_dir = Path(temp_checkpoint_dir)
                        checkpoint_path = temp_checkpoint_dir / "checkpoint.pt"
                        torch.save(
                            (
                                optimizer.state_dict(),
                                scheduler.state_dict(),
                                chunk_id,
                                run_name,
                            ),
                            checkpoint_path,
                        )
                        save_model(
                            model, hyperparameters, temp_checkpoint_dir / "model"
                        )
                        writer.add_hparams(
                            serialise_hparams(hyperparameters),
                            {
                                "Loss/test/epoch": chunk_test_loss,
                                "Loss/train/epoch": chunk_training_loss,
                            },
                            global_step=global_step,
                            run_name=(log_dir / run_name).absolute(),
                        )
                        train.report(
                            {
                                "chunk_test_loss": chunk_test_loss,
                                "chunk_training_loss": chunk_training_loss,
                            },
                            checkpoint=Checkpoint.from_directory(temp_checkpoint_dir),
                        )

                    scheduler.step()

        return inner(hyperparameters=hyperparameters, **hyperparameters)

    return train_with_ray


def load_or_create_state(
    device, log_dir, model_type, learning_rate, scheduler_gamma, **hyperparameters
) -> Tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    int,
    str,
]:
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            loaded_checkpoint_dir = Path(loaded_checkpoint_dir)
            model, hyperparameters = load_model(
                loaded_checkpoint_dir / "model", device=device
            )
            optimizer = Adam(model.parameters(), lr=learning_rate)

            optimizer_state, scheduler_state, start_chunk_id, run_name = torch.load(
                loaded_checkpoint_dir / "checkpoint.pt"
            )
            optimizer.load_state_dict(optimizer_state)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=scheduler_gamma
            )
            scheduler.load_state_dict(scheduler_state)
    else:
        run_name = get_next_run_name(log_dir)
        model = create_model(
            type=model_type,
            hyperparameters=hyperparameters,
            device=device,
        ).train()
        optimizer = Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=scheduler_gamma
        )
        start_chunk_id = 0

    return model, optimizer, scheduler, start_chunk_id, run_name


def write_histograms(
    model: torch.nn.Module,
    examples: List[Tuple[torch.Tensor, torch.Tensor]],
    writer: SummaryWriter,
    device: torch.device,
    global_step: int,
):
    edited_histograms, original_histograms = examples
    predicted_originals = model(edited_histograms.to(device))
    for i, (original, edited, predicted) in enumerate(
        zip(original_histograms, edited_histograms, predicted_originals)
    ):
        writer.add_figure(
            f"histogram_{i}",
            plot_histograms_in_2d(
                {
                    "original": original[0].numpy().squeeze(),
                    "edited": edited.cpu()[0].numpy().squeeze(),
                    "predicted": predicted.cpu()[0].numpy().squeeze(),
                }
            ),
            global_step=global_step,
        )
