import json

import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from torch import nn
import os
import pandas as pd
import matplotlib.pyplot as plt

from emulator.multisite.mlp import MLP
from emulator.multisite.tune_hyperparam import load_data

if __name__ == "__main__":
    device = torch.device("cpu")
    model_path = f"~/EMOD-calibration/emulator/multisite/trained_models/model_optim_multisite_1697157466"
    config_path = f"~/EMOD-calibration/emulator/multisite/configs/best_result_config_multisite_1697157466.json"
    with open(os.path.expanduser(config_path)) as fp:
        config = json.load(fp)

    mlp = MLP(
        l1=config["l1"],
        l2=config["l2"],
        l3=config["l3"],
        l4=config["l4"],
        l5=config["l5"],
        l6=config["l6"],
    )
    mlp.load_state_dict(torch.load(os.path.expanduser(model_path), map_location=device))
    mlp.eval()
    batch_size = 1
    train_loader, validation_loader, test_loader = load_data(batch_size=batch_size)

    with open("../output_dimensions_aggregate.json", "r") as content:
        dims = json.load(content)

    # Load min/max to rescale
    param_min = pd.read_csv("~/EMOD-calibration/simulations/download/param_mins.csv")
    param_max = pd.read_csv("~/EMOD-calibration/simulations/download/param_max.csv")

    # Write predictions to dataframe
    param_sets = []
    loss_vals = []

    num_batches = len(test_loader)
    loss_function = nn.MSELoss()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            individual_loss = []
            X, y, _ = data
            pred = mlp(X)
            param_sets.append(pd.DataFrame(X.cpu().numpy()).transpose())
            for output_type in dims.keys():
                dim_info = dims[output_type]
                begin_idx = dim_info["begin_idx"]
                end_idx = dim_info["end_idx"]
                # Extract correct data given output, calc loss against emulator
                head_loss = loss_function(pred[output_type], y[:, begin_idx:end_idx])
                individual_loss.append(head_loss)
            batch_loss = sum(individual_loss)
            loss_vals.append(batch_loss.item())

    param_df = pd.concat(param_sets, axis=1).transpose()
    param_df.columns = [
        "Falciparum_MSP_Variants",
        "Falciparum_Nonspecific_Types",
        "MSP1_Merozoite_Kill_Fraction",
        "Max_Individual_Infections",
        "Falciparum_PfEMP1_Variants",
        "Antigen_Switch_Rate",
        "Nonspecific_Antigenicity_Factor",
        "Site",
    ]

    loss_df = pd.DataFrame(loss_vals)
    loss_df.columns = ["loss"]
    top_10_loss_df = loss_df.sort_values(by="loss", ascending=False).head(100)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.ravel()
    for i, col in enumerate(param_df.columns[:7]):
        min_val = param_min.loc[param_min.iloc[:, 0] == col].iloc[:, 1].values[0]
        max_val = param_max.loc[param_min.iloc[:, 0] == col].iloc[:, 1].values[0]
        scaled_col = param_df[col] * (max_val - min_val) + min_val
        loss = loss_df["loss"]
        opacity = np.where(
            loss.index.isin(top_10_loss_df.index), 0.5, 0.005
        )  # Adjust opacity
        size = np.where(loss.index.isin(top_10_loss_df.index), 20, 5)  # Adjust opacity
        axes[i].scatter(scaled_col, loss, c=loss, alpha=opacity, cmap="viridis", s=size)
        if col == "Antigen_Switch_Rate":
            axes[i].set_xlim((0, 0.000001))
        if col == "Nonspecific_Antigenicity_Factor":
            axes[i].set_xlim((0, 0.00000001))
        axes[i].set_xlabel(col)  # Set x-axis label
        # axes[i].set_ylabel("loss")  # Set y-axis label
        axes[i].set_title(f"{col}")  # Set subplot title

        # Colorbar
        # Add a colorbar
        sm = ScalarMappable(cmap="viridis")
        sm.set_array([])  # Create an empty array to map colors to
        cbar = plt.colorbar(sm, ax=axes[i])
        cbar.set_label("Loss", rotation=270, labelpad=15)

    fig.delaxes(axes[7])
    fig.delaxes(axes[8])
    plt.tight_layout()  # Ensures subplots don't overlap
    plt.show()  # Display the plot
