"""
Stochastic gradient descent-based calibration.
"""

import torch
import pandas as pd
from torch import nn
import os
import json
import itertools
from scipy.stats import qmc
from torch.optim.lr_scheduler import StepLR
from calibration.config import site_metadata
from calibration.helpers import assemble_site_reference_data
from transfer_learning.helpers import get_latest_emulator
from transfer_learning.mlp import MLP
import numpy as np
import simulations.manifest as manifest
from simulations import params


def format_emulator_data_garki(emulator_outputs=None, months=None, dims=None):
    return generate_emulator_df(
        dims=dims,
        output_key="Parasitemia_2",
        channel_name="Smeared PfPR by Parasitemia and Age Bin",
        emulator_outputs=emulator_outputs,
        months=months,
    )


def generate_emulator_df(dims, output_key, channel_name, emulator_outputs, months):
    n_month = 12
    n_age = dims[output_key]["n_age"]
    n_density = dims[output_key]["n_density"]
    combinations = list(
        itertools.product(
            range(1, n_month + 1), range(1, n_age + 1), range(0, n_density)
        )
    )
    emulator_df = pd.DataFrame(
        combinations, columns=["Date", "Age Bin", "PfPR Bin"]
    ).sort_values(by=["Date", "Age Bin"])
    emulator_df["Counts"] = emulator_outputs[output_key][0].detach()
    emulator_df["Channel"] = channel_name
    emulator_df = emulator_df.set_index(["Channel", "Date", "Age Bin", "PfPR Bin"])
    emulator_df = emulator_df.loc[
        emulator_df.index.get_level_values("Date").isin(months)
    ]
    return emulator_df


def format_emulator_data_burkinafaso(emulator_outputs=None, months=None, dims=None):
    # Data are ordered by (month, age)
    # Create data frame and rebin/index like reference data
    parasitemia_df = generate_emulator_df(
        dims=dims,
        output_key="Parasitemia_1",
        channel_name="Smeared PfPR by Parasitemia and Age Bin",
        emulator_outputs=emulator_outputs,
        months=months,
    )

    gametocytemia_df = generate_emulator_df(
        dims=dims,
        output_key="Gametocytemia_1",
        channel_name="Smeared PfPR by Gametocytemia and Age Bin",
        emulator_outputs=emulator_outputs,
        months=months,
    )

    return pd.concat([gametocytemia_df, parasitemia_df])


def calculate_loss_incidence(
    emulator_outputs=None,
    reference_data=None,
    site=None,
    dims=None,
):
    # Format dataframe into Observations / Trials format for LL calculations
    emulator_outputs = emulator_outputs["Incidence"]
    reference_data = torch.tensor(reference_data, dtype=torch.float32).unsqueeze(0)
    loss_fn = nn.MSELoss()
    return loss_fn(emulator_outputs, reference_data)


def calculate_loss_prevalence(
    emulator_outputs=None, reference_data=None, site=None, dims=None
):
    emulator_outputs = emulator_outputs["PfPr"]
    reference_data = torch.tensor(reference_data, dtype=torch.float32).unsqueeze(0)
    loss_fn = nn.MSELoss()
    return loss_fn(emulator_outputs, reference_data)


def calculate_loss_parasitemia(
    emulator_outputs=None, reference_data=None, site=None, dims=None
):
    emulator_df = format_emulator_data_garki(
        emulator_outputs, months=site_metadata[site]["months"], dims=dims
    ).sort_values(by=["Age Bin", "Date"])
    reference_df = reference_data.sort_values(by=["Age Bin", "Date"])
    loss_fn = nn.MSELoss()
    emulator_values = torch.tensor(emulator_df["Counts"].values).unsqueeze(0)
    reference_values = torch.tensor(
        reference_df["Counts"].values, dtype=torch.float32
    ).unsqueeze(0)
    return loss_fn(emulator_values, reference_values)


def calculate_loss_parasitemia_gametocytemia(
    emulator_outputs=None, reference_data=None, site=None, dims=None
):
    emulator_df = format_emulator_data_burkinafaso(
        emulator_outputs, months=site_metadata[site]["months"], dims=dims
    ).sort_values(by=["Age Bin", "Date"])
    reference_df = reference_data.sort_values(by=["Age Bin", "Date"]).sort_values(
        by=["Age Bin", "Date"]
    )
    loss_fn = nn.MSELoss()
    emulator_values = torch.tensor(emulator_df["Counts"].values).unsqueeze(0)
    reference_values = torch.tensor(
        reference_df["Counts"].values, dtype=torch.float32
    ).unsqueeze(0)
    return loss_fn(emulator_values, reference_values)


def calculate_joint_loss(
    mlp, params, site_tokens, data_dict, min_values, max_values, sim_df, exclude_site
):
    loss_functions = {
        "namawala_1991": calculate_loss_prevalence,
        "ndiop_1993": calculate_loss_incidence,
        "dielmo_1990": calculate_loss_incidence,
        "matsari_1970": calculate_loss_parasitemia,
        "rafin_marke_1970": calculate_loss_parasitemia,
        "sugungum_1970": calculate_loss_parasitemia,
        "dapelogo_2007": calculate_loss_parasitemia_gametocytemia,
        "laye_2007": calculate_loss_parasitemia_gametocytemia,
    }
    site_losses = []

    for site in loss_functions.keys():
        if site == exclude_site:
            site_losses.append(0.0)
        else:
            outputs_for_site = site_metadata[site]["emulator_outcomes"]
            lf = loss_functions[site]
            token = site_tokens[site].iloc[0]
            site_df = sim_df[sim_df.Site == token]
            site_params = (
                site_df.filter(regex="eir|has_interventions", axis=1).iloc[0].values
            )
            inputs = torch.cat((params, torch.tensor(site_params))).unsqueeze(0)
            inputs = normalize(inputs, min_values, max_values, num_features=19)
            inputs = inputs.to(torch.float32)
            outputs = mlp(inputs)
            site_loss = lf(
                emulator_outputs=dict(
                    zip(outputs_for_site, [outputs[o] for o in outputs_for_site])
                ),
                reference_data=data_dict[site],
                site=site,
                dims=dims,
            )
            site_losses.append(site_loss)

    return sum(site_losses)


def project_to_positive(params, integer_params, params_df):
    with torch.no_grad():
        params.data = torch.clamp(
            params,
            min=torch.tensor(params_df["min"].values, dtype=torch.float).squeeze(0),
            max=torch.tensor(params_df["max"].values, dtype=torch.float).squeeze(0),
        )
        params.data[integer_params] = torch.round(params.data[integer_params])


def trunc(values, decs=0):
    return np.trunc(values * 10**decs) / (10**decs)


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
    _, data_dict = assemble_site_reference_data()
    device = "cpu"
    path = os.path.expanduser("~/EMOD-calibration/transfer_learning/scaled_df.pkl")
    sim_df = pd.read_pickle(path)

    for exclude_site in params.sites:
        model_path, config_path = get_latest_emulator(exclude_site=exclude_site)
        lrs = [0.1, 0.01, 0.001]
        with open(os.path.expanduser(config_path)) as fp:
            config = json.load(fp)

        site_tokens = pd.read_csv(
            "~/EMOD-calibration/transfer_learning/site_tokens.csv", index_col=0
        )

        with open(
            os.path.expanduser(
                "~/EMOD-calibration/transfer_learning/output_dimensions_aggregate.json"
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
        mlp.load_state_dict(
            torch.load(os.path.expanduser(model_path), map_location=device)
        )
        mlp.eval()

        train_params = [
            "Falciparum_MSP_Variants",
            "Falciparum_Nonspecific_Types",
            "MSP1_Merozoite_Kill_Fraction",
            "Max_Individual_Infections",
            "Falciparum_PfEMP1_Variants",
            "Antigen_Switch_Rate",
            "Nonspecific_Antigenicity_Factor",
        ]

        n_epochs = 200
        n_trials = 200

        # Normalization constants
        min_values = pd.read_csv(
            os.path.expanduser(
                "~/EMOD-calibration/simulations/download/param_mins_transfer.csv"
            ),
            index_col=0,
        )
        min_values = torch.tensor(
            min_values.T.values, dtype=torch.float32, device=device
        )
        max_values = pd.read_csv(
            os.path.expanduser(
                "~/EMOD-calibration/simulations/download/param_max_transfer.csv"
            ),
            index_col=0,
        )
        max_values = torch.tensor(
            max_values.T.values, dtype=torch.float32, device=device
        )
        integer_params = [0, 1, 3, 4]
        for lr in lrs:
            print(f">>> Beginning optimization with initial learning rate: {lr}")
            optim_losses = []

            key_path = manifest.parameters_key_path
            params_df = pd.read_csv(key_path)

            # parameters to generate LHS samples
            sweep_ids = [2, 6, 7, 8, 11, 12, 14]

            # parameters that must be integers
            params_df = params_df[params_df.parameter_id.isin(sweep_ids)]
            params_df = params_df.sort_values(
                by="parameter_name", key=lambda x: [train_params.index(v) for v in x]
            )
            # set up sampler
            dim = len(sweep_ids)
            sampler = qmc.LatinHypercube(d=dim)

            # sample and scale to bounds
            sample = sampler.random(n=n_trials)
            lower_bounds = [
                np.log10(min_val)
                if (
                    parameter_name == "Antigen_Switch_Rate"
                    or parameter_name == "Nonspecific_Antigenicity_Factor"
                )
                else min_val
                for min_val, parameter_name in zip(
                    params_df["min"].values, params_df["parameter_name"]
                )
            ]
            upper_bounds = [
                np.log10(max_val)
                if (
                    parameter_name == "Antigen_Switch_Rate"
                    or parameter_name == "Nonspecific_Antigenicity_Factor"
                )
                else max_val
                for max_val, parameter_name in zip(
                    params_df["max"].values, params_df["parameter_name"]
                )
            ]
            sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
            sample_scaled = pd.DataFrame(
                sample_scaled, columns=params_df["parameter_name"]
            )
            sample_scaled["Antigen_Switch_Rate"] = np.power(
                10, sample_scaled["Antigen_Switch_Rate"]
            ).apply(
                lambda x: np.format_float_positional(x, precision=2, fractional=False)
            )

            sample_scaled["Nonspecific_Antigenicity_Factor"] = np.power(
                10, sample_scaled["Nonspecific_Antigenicity_Factor"]
            ).apply(
                lambda x: np.format_float_positional(x, precision=2, fractional=False)
            )

            sample_scaled["MSP1_Merozoite_Kill_Fraction"] = sample_scaled[
                "MSP1_Merozoite_Kill_Fraction"
            ].apply(
                lambda x: np.format_float_positional(x, precision=2, fractional=False)
            )

            # convert parameters that need to be integers
            sample_scaled = sample_scaled.astype(
                {
                    "Falciparum_MSP_Variants": int,
                    "Falciparum_Nonspecific_Types": int,
                    "Falciparum_PfEMP1_Variants": int,
                    "Max_Individual_Infections": int,
                }
            )
            sample_scaled = sample_scaled.reindex(columns=train_params)

            for i, param_set in enumerate(sample_scaled.values):
                print(f">>> Parameter set {i + 1} of {n_trials}")
                params = torch.tensor(
                    np.float32(param_set), dtype=torch.float, requires_grad=True
                )
                optimizer = torch.optim.Adam([params], lr=lr)
                scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
                for j in range(n_epochs):
                    loss = calculate_joint_loss(
                        mlp,
                        params,
                        site_tokens,
                        data_dict,
                        min_values,
                        max_values,
                        sim_df,
                        exclude_site,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # Project parameters onto positive orthant
                    project_to_positive(params, integer_params, params_df)

                    scheduler.step()
                row = pd.concat(
                    [
                        pd.Series(params.detach()),
                        pd.Series({"loss": loss.item()}),
                        pd.Series({"lr": lr}),
                    ]
                )
                optim_losses.append(row)

            optim_df = pd.concat(optim_losses, axis=1).transpose()
            optim_df.to_csv(
                os.path.expanduser(
                    f"~/EMOD-calibration/transfer_learning/calibration/data/exclude_{exclude_site}/sgd_loss_df_{lr}.csv"
                )
            )
