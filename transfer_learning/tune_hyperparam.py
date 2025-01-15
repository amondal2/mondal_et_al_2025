"""
Script to tune hyperparameters for four-head MLP.
"""
import pandas as pd
import torch
import json
import time
import os

from ray.tune.schedulers import ASHAScheduler
from torch import nn
from torch.optim.lr_scheduler import StepLR

from transfer_learning.dataset import EmulatorDataset
from transfer_learning.mlp import MLP
from torch.utils.data import DataLoader, random_split
from emulator.multisite.config_hyperparam import config
from simulations.params import sites
from ray.air import Checkpoint, session
from ray import tune


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


def load_data(batch_size=32, exclude_site=None):
    path = os.path.expanduser("~/EMOD-calibration/transfer_learning/scaled_df.pkl")
    # Load data
    emod_data = EmulatorDataset(input_file=path, exclude_site=exclude_site)
    data_samples = len(emod_data)
    train_size = int(0.80 * data_samples)
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


def train_loop(
    train_loader,
    mlp,
    loss_function,
    optimizer,
    scheduler,
    dims,
    device,
    max_values,
    min_values,
):
    # Iterate over the DataLoader for training data
    mlp.train()
    num_batches = len(train_loader)
    train_loss_individual = dict.fromkeys(dims.keys(), 0.0)
    train_loss_total = 0.0
    for i, data in enumerate(train_loader, 0):
        # Get inputs
        inputs, targets, _ = data[0].to(device), data[1].to(device), data[2]
        inputs = normalize(inputs, min_values, max_values, num_features=19)
        # Zero the gradients
        optimizer.zero_grad()
        # Perform forward pass
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
    return train_loss_total, train_loss_individual


def validation_loop(
    dataloader, model, loss_function, dims, device, max_values, min_values
):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    val_loss_total = 0.0
    val_loss_individual = dict.fromkeys(dims.keys(), 0.0)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y, _ in dataloader:
            X = X.to(device)
            X = normalize(X, min_values, max_values, num_features=19)
            y = y.to(device)
            pred = model(X)
            individual_loss = []
            for output_type in dims.keys():
                dim_info = dims[output_type]
                begin_idx = dim_info["begin_idx"]
                end_idx = dim_info["end_idx"]
                head_loss = loss_function(pred[output_type], y[:, begin_idx:end_idx])
                # Extract correct data given output, calc loss against emulator
                val_loss_individual[output_type] = (
                    val_loss_individual[output_type] + head_loss.item()
                )
                individual_loss.append(head_loss)
            total_loss = torch.sum(torch.stack(individual_loss))
            val_loss_total = val_loss_total + total_loss.item()

    val_loss_total = val_loss_total / num_batches
    val_loss_individual = {k: v / num_batches for k, v in val_loss_individual.items()}
    print(f"Validation Error: Avg loss: {val_loss_total:>8f} \n")
    return val_loss_total, val_loss_individual


def train_mlp(
    config,
    loss_function,
    device="cpu",
    dims=None,
    exclude_site=None,
    max_values=None,
    min_values=None,
):
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
    optimizer = torch.optim.Adam(
        mlp.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler_optim = StepLR(optimizer, step_size=10, gamma=0.1)

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        mlp.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    train_loader, val_loader, test_loader = load_data(
        batch_size=config["batch_size"], exclude_site=exclude_site
    )

    for t in range(start_epoch, 10):
        print(
            f"Epoch {t + 1}, excluding {exclude_site}\n-------------------------------"
        )
        avg_trainloss, individual_trainloss = train_loop(
            train_loader,
            mlp,
            loss_function,
            optimizer,
            scheduler_optim,
            dims,
            device,
            max_values,
            min_values,
        )
        avg_vloss, individual_valloss = validation_loop(
            val_loader, mlp, loss_function, dims, device, max_values, min_values
        )

        checkpoint_data = {
            "epoch": t,
            "net_state_dict": mlp.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {
                "val_loss": avg_vloss,
                "train_loss": avg_trainloss,
                "val_loss_incidence": individual_valloss["Incidence"],
                "val_loss_gam1": individual_valloss["Gametocytemia_1"],
                "val_loss_par1": individual_valloss["Parasitemia_1"],
                "val_loss_par2": individual_valloss["Parasitemia_2"],
                "val_loss_pfpr": individual_valloss["PfPr"],
            },
            checkpoint=checkpoint,
        )

    print(">>> Finished training ...\n")


if __name__ == "__main__":
    max_num_epochs = 50
    num_samples = 50

    with open(
        os.path.expanduser(
            "~/EMOD-calibration/transfer_learning/output_dimensions_aggregate.json"
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

    min_values = pd.read_csv(
        os.path.expanduser(
            "~/EMOD-calibration/simulations/download/param_mins_transfer.csv"
        ),
        index_col=0,
    )
    min_values = torch.tensor(min_values.T.values, dtype=torch.float32, device=device)
    max_values = pd.read_csv(
        os.path.expanduser(
            "~/EMOD-calibration/simulations/download/param_max_transfer.csv"
        ),
        index_col=0,
    )
    max_values = torch.tensor(max_values.T.values, dtype=torch.float32, device=device)

    for site in sites:
        loss_function = nn.MSELoss()
        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=10,
            reduction_factor=2,
        )
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    train_mlp,
                    loss_function=loss_function,
                    device=device,
                    dims=dims,
                    exclude_site=site,
                    max_values=max_values,
                    min_values=min_values,
                ),
                resources={"cpu": 32, "gpu": 1},
            ),
            tune_config=tune.TuneConfig(
                metric="val_loss",
                mode="min",
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            param_space=config,
        )
        results = tuner.fit()
        best_result = results.get_best_result("val_loss", "min")

        print(f"Best trial config: {best_result.config}")
        print(f"Best trial final validation loss: {best_result.metrics['val_loss']}")
        print(f"Best trial final training loss: {best_result.metrics['train_loss']}")

        best_trained_model = MLP(
            l1=best_result.config["l1"],
            l2=best_result.config["l2"],
            l3=best_result.config["l3"],
            l4=best_result.config["l4"],
            l5=best_result.config["l5"],
            l6=best_result.config["l6"],
            l7=best_result.config["l7"],
            l8=best_result.config["l8"],
            l9=best_result.config["l9"],
            l10=best_result.config["l10"],
            l11=best_result.config["l11"],
            dropout_prob=best_result.config["dropout_prob"],
        ).to(device)

        checkpoint = best_result.checkpoint.to_dict()
        best_trained_model.load_state_dict(checkpoint["net_state_dict"])
        best_trained_model.eval()

        # Test model on unseen data
        train_loader, val_loader, test_loader = load_data(
            batch_size=best_result.config["batch_size"], exclude_site=site
        )

        num_batches = len(test_loader)
        test_loss_total = 0.0
        test_loss_individual = dict.fromkeys(dims.keys(), 0.0)

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y, _ in test_loader:
                X = X.to(device)
                X = normalize(X, min_values, max_values, num_features=19)
                y = y.to(device)
                pred = best_trained_model(X)
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
        best_result.metrics["test_set_loss"] = test_loss_total

        # Write everything to disk
        timestamp = int(time.time())
        config_path = f"~/EMOD-calibration/transfer_learning/configs/exclude_{site}/best_result_config_multisite_{timestamp}.json"
        metrics_path = f"~/EMOD-calibration/transfer_learning/metrics/exclude_{site}/best_result_metrics_multisite_{timestamp}.json"
        with open(os.path.expanduser(config_path), "w") as fp_config:
            json.dump(best_result.config, fp_config)

        with open(os.path.expanduser(metrics_path), "w") as fp_metrics:
            json.dump(best_result.metrics, fp_metrics)
