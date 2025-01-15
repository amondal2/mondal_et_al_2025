import itertools
import time

import pandas as pd
import os
import json
import numpy as np
import seaborn as sns
import calendar
from matplotlib.font_manager import FontProperties
from scipy.stats import binom
from calibration.config import (
    site_metadata,
    namawala_age_bins,
    senegal_age_bins,
)
from calibration.helpers import (
    get_garki_reference,
    assemble_site_reference_data,
    get_bf_reference,
)
from emulator.helpers import get_latest_emulator, site_names
from emulator.multisite.mlp import MLP
from torch import nn
import torch
import matplotlib.pyplot as plt


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
    device = torch.device("cpu")
    model_path, config_path = get_latest_emulator()
    with open(os.path.expanduser(config_path)) as fp:
        config = json.load(fp)

    site_tokens = pd.read_csv(
        "~/EMOD-calibration/emulator/site_tokens.csv", index_col=0
    )

    with open(
        os.path.expanduser(
            "~/EMOD-calibration/emulator/output_dimensions_aggregate.json"
        ),
        "r",
    ) as content:
        dims = json.load(content)

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
    )
    mlp.load_state_dict(torch.load(os.path.expanduser(model_path), map_location=device))
    mlp.eval()

    # Load simulation data
    path = os.path.expanduser("~/EMOD-calibration/emulator/scaled_df.pkl")
    sim_df = pd.read_pickle(path)

    # Normalization constants
    min_values = pd.read_csv(
        os.path.expanduser("~/EMOD-calibration/simulations/download/param_mins.csv"),
        index_col=0,
    )
    min_values = torch.tensor(min_values.T.values, dtype=torch.float32, device=device)
    max_values = pd.read_csv(
        os.path.expanduser("~/EMOD-calibration/simulations/download/param_max.csv"),
        index_col=0,
    )
    max_values = torch.tensor(max_values.T.values, dtype=torch.float32, device=device)

    # Split data into unique sets of parameters
    train_params = [
        "Falciparum_MSP_Variants",
        "Falciparum_Nonspecific_Types",
        "MSP1_Merozoite_Kill_Fraction",
        "Max_Individual_Infections",
        "Falciparum_PfEMP1_Variants",
        "Antigen_Switch_Rate",
        "Nonspecific_Antigenicity_Factor",
        "Site",
    ]
    dfs = [y for x, y in sim_df.groupby(train_params, as_index=False)]

    loss_rows = []
    site_dict = dict(zip(site_tokens.iloc[0].values, site_tokens.columns))
    rows = []
    with torch.no_grad():
        for df in dfs:
            sim_means = df.iloc[:, 9:].mean().rename("mean_{}".format)
            site = site_dict[df.Site.iloc[0]]

            # Eval emulator
            input_params = df.iloc[:, :8].iloc[0]
            inputs = torch.tensor(input_params.values, dtype=torch.float32).reshape(
                1, 8
            )
            inputs = normalize(inputs, min_values, max_values, num_features=7)
            start = time.time()
            outputs = mlp(inputs)
            end = time.time()
            row = pd.Series({"runtime": end - start, "site": site})
            rows.append(row)
    df = pd.concat(rows, axis=1).transpose()
    df_path = os.path.expanduser(
        "~/EMOD-calibration/emulator/multisite/emulator_runtimes.csv"
    )
    df.to_csv(df_path)
