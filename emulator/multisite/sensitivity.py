"""
Retraining MLP for more epochs after hyperparameter selection.
"""
import torch
import os
import json
import time
import pandas as pd
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

from emulator.dataset import EmulatorDataset
from emulator.helpers import get_latest_emulator
from emulator.multisite.mlp import MLP


def load_data(batch_size=32, site=None, dataset_fraction=0.8):
    path = os.path.expanduser("~/EMOD-calibration/emulator/scaled_df.pkl")
    # Load data
    emod_data = EmulatorDataset(input_file=path, site=site)
    data_samples = len(emod_data)
    train_size = int(dataset_fraction * data_samples)
    validation_size = int(0.10 * data_samples)
    test_size = data_samples - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = random_split(
        emod_data,
        [train_size, validation_size, test_size],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=5
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=5,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=5,
    )

    return train_loader, validation_loader, test_loader


def normalize(batch, min_values, max_values, num_features=7):
    """
    Normalize the first `num_features` columns of a batch of data using pre-defined
    minimum and maximum values.

    Args:
        batch (torch.Tensor): Input batch of data to be normalized.
        min_values (list or torch.Tensor): Minimum values for normalization.
        max_values (list or torch.Tensor): Maximum values for normalization.
        num_features (int): Number of features to normalize.

    Returns:
        torch.Tensor: Normalized batch of data.
    """

    # Normalize the first `num_features` columns of the batch
    normalized_batch = batch.clone()
    normalized_batch[:, :num_features] = (batch[:, :num_features] - min_values) / (
        max_values - min_values
    )

    return normalized_batch


if __name__ == "__main__":
    n_epochs = 100
    loss_function = nn.MSELoss()
    with open(
        os.path.expanduser(
            "~/EMOD-calibration/emulator/output_dimensions_aggregate.json"
        ),
        "r",
    ) as content:
        dims = json.load(content)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model_path, config_path = get_latest_emulator()

    with open(os.path.expanduser(config_path)) as fp:
        config = json.load(fp)
    rows = []
    for rep in range(50):
        for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            mlp = MLP(
                l1=config["l1"],
                l2=config["l2"],
                l3=config["l3"],
                l4=config["l4"],
                l5=config["l5"],
                l6=config["l6"],
                l7=config["l7"],
                l8=config["l8"],
                l9=config["l9"],
                l10=config["l10"],
                l11=config["l11"],
                dropout_prob=config["dropout_prob"],
            ).to(device)
            mlp.train()

            optimizer = torch.optim.Adam(
                mlp.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            train_loader, val_loader, test_loader = load_data(
                batch_size=config["batch_size"], dataset_fraction=frac
            )
            best_val_loss = 1000000

            min_values = pd.read_csv(
                os.path.expanduser(
                    "~/EMOD-calibration/simulations/download/param_mins.csv"
                ),
                index_col=0,
            )
            min_values = torch.tensor(
                min_values.T.values, dtype=torch.float32, device=device
            )
            max_values = pd.read_csv(
                os.path.expanduser(
                    "~/EMOD-calibration/simulations/download/param_max.csv"
                ),
                index_col=0,
            )
            max_values = torch.tensor(
                max_values.T.values, dtype=torch.float32, device=device
            )

            for epoch in range(n_epochs):
                print(f"EPOCH >>> {epoch + 1}")
                num_batches = len(train_loader)
                train_loss_individual = dict.fromkeys(dims.keys(), 0.0)
                train_loss_total = 0.0
                for i, data in enumerate(train_loader, 0):
                    # Get inputs
                    inputs, targets, _ = data[0].to(device), data[1].to(device), data[2]
                    # Convert min_values and max_values to tensors if they are not already

                    # Zero the gradients
                    optimizer.zero_grad()
                    # Perform forward pass
                    inputs = normalize(inputs, min_values, max_values, num_features=7)
                    outputs = mlp(inputs)

                    # Compute loss across heads
                    individual_loss = []
                    for output_type in dims.keys():
                        dim_info = dims[output_type]
                        begin_idx = dim_info["begin_idx"]
                        end_idx = dim_info["end_idx"]
                        loss_multiplier = dim_info["loss_multiplier"]

                        # Extract correct data given output, calc loss against emulator
                        head_loss = loss_multiplier * loss_function(
                            outputs[output_type],
                            targets[:, begin_idx:end_idx],
                        )
                        individual_loss.append(head_loss)
                        train_loss_individual[output_type] = (
                            train_loss_individual[output_type] + head_loss.item()
                        )
                    total_loss = torch.sum(torch.stack(individual_loss))
                    train_loss_total = train_loss_total + total_loss.item()

                    # Perform backward pass
                    total_loss.backward()

                    # Perform optimization
                    optimizer.step()
                scheduler.step()

                # Get averages
                train_loss_total = train_loss_total / num_batches
                train_loss_individual = {
                    k: v / num_batches for k, v in train_loss_individual.items()
                }
                print(f"Train Error: Avg loss: {train_loss_total:>8f} \n")
                mlp.eval()
                num_batches = len(val_loader)
                val_loss_total = 0.0
                val_loss_individual = dict.fromkeys(dims.keys(), 0.0)

                # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
                # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
                with torch.no_grad():
                    for X, y, _ in val_loader:
                        X = X.to(device)
                        X = normalize(X, min_values, max_values, num_features=7)
                        y = y.to(device)
                        pred = mlp(X)
                        individual_loss = []
                        for output_type in dims.keys():
                            dim_info = dims[output_type]
                            begin_idx = dim_info["begin_idx"]
                            end_idx = dim_info["end_idx"]
                            head_loss = loss_function(
                                pred[output_type], y[:, begin_idx:end_idx]
                            )
                            # Extract correct data given output, calc loss against emulator
                            val_loss_individual[output_type] = (
                                val_loss_individual[output_type] + head_loss.item()
                            )
                            individual_loss.append(head_loss)
                        total_loss = torch.sum(torch.stack(individual_loss))
                        val_loss_total = val_loss_total + total_loss.item()

                val_loss_total = val_loss_total / num_batches
                val_loss_individual = {
                    k: v / num_batches for k, v in val_loss_individual.items()
                }
                print(f"Validation Error: Avg loss: {val_loss_total:>8f} \n")
                if val_loss_total < best_val_loss:
                    best_val_loss = val_loss_total
                else:
                    break

            print(">>> TESTING")
            num_batches = len(test_loader)
            test_loss_total = 0.0
            test_loss_individual = dict.fromkeys(dims.keys(), 0.0)

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
            with torch.no_grad():
                for X, y, _ in test_loader:
                    X = X.to(device)
                    X = normalize(X, min_values, max_values, num_features=7)
                    y = y.to(device)
                    pred = mlp(X)
                    individual_loss = []
                    for output_type in dims.keys():
                        dim_info = dims[output_type]
                        begin_idx = dim_info["begin_idx"]
                        end_idx = dim_info["end_idx"]

                        # Extract correct data given output, calc loss against emulator
                        head_loss = loss_function(
                            pred[output_type], y[:, begin_idx:end_idx]
                        )
                        test_loss_individual[output_type] = (
                            test_loss_individual[output_type] + head_loss.item()
                        )
                        individual_loss.append(head_loss)
                    total_loss = torch.sum(torch.stack(individual_loss))
                    test_loss_total = test_loss_total + total_loss.item()

            test_loss_total = test_loss_total / num_batches
            test_loss_individual = {
                k: v / num_batches for k, v in test_loss_individual.items()
            }
            print("Best trial test set loss: {}".format(test_loss_total))
            print(f"Individual task losses: {test_loss_individual}")
            df_row = pd.Series(
                {
                    "frac": frac,
                    "rep": int(rep),
                    "loss": test_loss_total,
                }
            )
            rows.append(df_row)

    loss_df = pd.concat(rows, axis=1).transpose()
    loss_df.to_csv(
        os.path.expanduser(f"~/EMOD-calibration/emulator/multisite/sensitivity_df.csv")
    )
