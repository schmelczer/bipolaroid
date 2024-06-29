import logging
from typing import Any, Dict, List
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.optim import Adam
from tqdm.notebook import tqdm
from visualisation import plot_histograms_in_2d
from models import create_model
import torch
from utils import serialise_hparams
from .get_data_loader import get_data_loader

EPSILON = 1e-5


def train(
    hyperparameters: Dict[str, Any],
    train_data_paths: List[Path],
    test_data_paths: List[Path],
    log_dir: Path,
    use_tqdm: bool,
    device: torch.device,
    model_type: str,
    learning_rate: float,
    scheduler_gamma: float,
    num_epochs: int,
    **_,
) -> torch.nn.Module:
    train_data_loader = get_data_loader(train_data_paths, **hyperparameters)
    test_data_loader = get_data_loader(test_data_paths, **hyperparameters)

    with SummaryWriter(log_dir) as writer:
        model = create_model(
            type=model_type,
            hyperparameters=hyperparameters,
            device=device,
        ).train()
        writer.add_graph(model, next(iter(train_data_loader))[0].to(device))

        optimizer = Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=scheduler_gamma
        )
        loss_function = torch.nn.KLDivLoss(reduction="batchmean").to(device)

        for epoch in range(num_epochs):
            epoch_loss = 0
            writer.add_scalar("Actual learning rate", scheduler.get_last_lr()[0], epoch)
            for batch_id, (edited_histogram, original_histogram) in enumerate(
                tqdm(train_data_loader, desc=f"Epoch {epoch}", unit="batch")
                if use_tqdm
                else train_data_loader
            ):
                optimizer.zero_grad()
                predicted_original = model(edited_histogram.to(device))
                loss = loss_function(
                    torch.log(predicted_original + EPSILON),
                    original_histogram.to(device),
                )

                epoch_loss += loss.item()
                writer.add_scalar(
                    "Loss/train/batch",
                    loss,
                    global_step=epoch * len(train_data_loader) + batch_id,
                )
                loss.backward()
                optimizer.step()

            logging.info(f"Epoch {epoch} train loss: {epoch_loss}")
            with torch.no_grad():
                model.eval()
                loader = iter(test_data_loader)
                edited_histogram, original_histogram = next(loader)
                predicted_original = model(edited_histogram.to(device))
                writer.add_figure(
                    "histogram",
                    plot_histograms_in_2d(
                        {
                            "original": original_histogram[0].numpy().squeeze(),
                            "edited": edited_histogram.cpu()[0].numpy().squeeze(),
                            "predicted": predicted_original.cpu()[0].numpy().squeeze(),
                        }
                    ),
                    epoch,
                )

                epoch_test_loss = 0
                for batch_id, (edited_histogram, original_histogram) in enumerate(
                    test_data_loader
                ):
                    predicted_original = model(edited_histogram.to(device))
                    epoch_test_loss += loss_function(
                        torch.log(predicted_original + EPSILON),
                        original_histogram.to(device),
                    ).item()
            writer.add_hparams(
                serialise_hparams(hyperparameters),
                {
                    "Loss/test/epoch": epoch_test_loss,
                    "Loss/train/epoch": epoch_loss,
                },
                global_step=epoch,
                run_name=log_dir.absolute(),
            )
            logging.info(f"Epoch {epoch} test loss: {epoch_test_loss}")

            model.train()
            scheduler.step()
        return model
