"""
Calibration to reference data using likelihood-based approach.
Calibrate across sites using joint likelihood.
"""

import pandas as pd
import os
import json
import numpy as np
import glob
from calibration.config import (
    site_metadata,
)
from calibration.helpers import (
    beta_binomial_likelihood,
    gamma_poisson_likelihood,
    dirichlet_multinomial_likelihood,
    calculate_observations,
    assemble_site_reference_data,
)
from emulator.helpers import get_latest_emulator, site_names
from emulator.multisite.mlp import MLP
import torch
import itertools


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


def calculate_likelihood_incidence(
    emulator_outputs=None,
    reference_data=None,
    site=None,
    dims=None,
):
    # Format dataframe into Observations / Trials format for LL calculations
    emulator_outputs = emulator_outputs["Incidence"]
    sim_pop = 1000
    site_pop = site_metadata[site]["population_by_age_bin"]
    emulator_observations = pd.Series(emulator_outputs.tolist()[0]) * sim_pop
    reference_observations = pd.Series(reference_data) * site_pop
    emulator_df = pd.DataFrame(
        {"Trials": sim_pop, "Observations": emulator_observations}
    )
    reference_df = pd.DataFrame(
        {"Trials": site_pop, "Observations": reference_observations}
    )
    likelihood = gamma_poisson_likelihood(emulator_df, reference_df)
    return {"ll": likelihood, "penalty_type": "gamma-poisson"}


def calculate_likelihood_prevalence(
    emulator_outputs=None, reference_data=None, site=None, dims=None
):
    # Format dataframe into Observations / Trials format for LL calculations
    emulator_outputs = emulator_outputs["PfPr"]
    sim_pop = 1000
    site_pop = site_metadata[site]["population_by_age_bin"]
    emulator_observations = pd.Series(emulator_outputs.tolist()[0]) * sim_pop
    reference_observations = pd.Series(reference_data) * site_pop
    emulator_df = pd.DataFrame(
        {"Trials": sim_pop, "Observations": emulator_observations}
    )
    reference_df = pd.DataFrame(
        {"Trials": site_pop, "Observations": reference_observations}
    )
    likelihood = beta_binomial_likelihood(emulator_df, reference_df)
    return {"ll": likelihood, "penalty_type": "beta-binomial"}


def format_emulator_data_garki(emulator_outputs=None, months=None, dims=None):
    return generate_trial_obs_df(
        dims=dims,
        output_key="Parasitemia_2",
        channel_name="Smeared PfPR by Parasitemia and Age Bin",
        emulator_outputs=emulator_outputs,
        months=months,
    )


def generate_trial_obs_df(dims, output_key, channel_name, emulator_outputs, months):
    sim_pop = 1000
    n_month = 12
    n_age = dims[output_key]["n_age"]
    n_density = dims[output_key]["n_density"]
    combinations = list(
        itertools.product(
            range(1, n_month + 1), range(1, n_age + 1), range(0, n_density)
        )
    )
    likelihood_df = pd.DataFrame(
        combinations, columns=["Date", "Age Bin", "PfPR Bin"]
    ).sort_values(by=["Date", "Age Bin"])
    likelihood_df["Counts"] = sim_pop * emulator_outputs[output_key][0]
    likelihood_df["Channel"] = channel_name
    likelihood_df = likelihood_df.set_index(["Channel", "Date", "Age Bin", "PfPR Bin"])
    likelihood_df = likelihood_df.loc[
        likelihood_df.index.get_level_values("Date").isin(months)
    ]
    trials_obs_df = pd.DataFrame()
    trials_obs_df["Trials"] = likelihood_df.groupby(
        ["Channel", "Date", "Age Bin"]
    ).apply(lambda x: sim_pop)
    trials_obs_df["Observations"] = likelihood_df.groupby(
        ["Channel", "Date", "Age Bin"]
    ).apply(lambda x: calculate_observations(x))
    return trials_obs_df.dropna()


def format_emulator_data_burkinafaso(emulator_outputs=None, months=None, dims=None):
    # Data are ordered by (month, age)
    # Create data frame and rebin/index like reference data
    parasitemia_df = generate_trial_obs_df(
        dims=dims,
        output_key="Parasitemia_1",
        channel_name="Smeared PfPR by Parasitemia and Age Bin",
        emulator_outputs=emulator_outputs,
        months=months,
    )

    gametocytemia_df = generate_trial_obs_df(
        dims=dims,
        output_key="Gametocytemia_1",
        channel_name="Smeared PfPR by Gametocytemia and Age Bin",
        emulator_outputs=emulator_outputs,
        months=months,
    )

    return pd.concat([gametocytemia_df, parasitemia_df])


def calculate_likelihood_parasitemia(
    emulator_outputs=None, reference_data=None, site=None, dims=None
):
    emulator_df = format_emulator_data_garki(
        emulator_outputs, months=site_metadata[site]["months"], dims=dims
    )
    likelihood = dirichlet_multinomial_likelihood(
        emulator_df=emulator_df,
        reference_df=reference_data,
        n_categories=dims["Parasitemia_2"]["n_density"],
    )
    return {"ll": likelihood, "penalty_type": "dirichlet-multinomial"}


def calculate_likelihood_parasitemia_gametocytemia(
    emulator_outputs=None, reference_data=None, site=None, dims=None
):
    emulator_df = format_emulator_data_burkinafaso(
        emulator_outputs, months=site_metadata[site]["months"], dims=dims
    )
    likelihood = dirichlet_multinomial_likelihood(
        emulator_df=emulator_df,
        reference_df=reference_data,
        n_categories=dims["Parasitemia_1"]["n_density"],
    )
    return {"ll": likelihood, "penalty_type": "dirichlet-multinomial"}


def calculate_joint_ll(mlp, params, site_tokens, data_dict, min_values, max_values):
    ll_functions = {
        "namawala_1991": calculate_likelihood_prevalence,
        "ndiop_1993": calculate_likelihood_incidence,
        "dielmo_1990": calculate_likelihood_incidence,
        "matsari_1970": calculate_likelihood_parasitemia,
        "rafin_marke_1970": calculate_likelihood_parasitemia,
        "sugungum_1970": calculate_likelihood_parasitemia,
        "dapelogo_2007": calculate_likelihood_parasitemia_gametocytemia,
        "laye_2007": calculate_likelihood_parasitemia_gametocytemia,
    }
    site_losses = []

    for site in ll_functions.keys():
        outputs_for_site = site_metadata[site]["emulator_outcomes"]
        lf = ll_functions[site]
        token = site_tokens[site].iloc[0]
        inputs = torch.cat((params, torch.tensor([token]))).unsqueeze(0)
        inputs = normalize(inputs, min_values, max_values, num_features=7)
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

    return sum([x["ll"] for x in site_losses])


def concatenate_csv_files(directory_path):
    # Get a list of all CSV files in the directory that start with 'sgd_loss'
    csv_files = glob.glob(os.path.join(directory_path, "sgd_loss*.csv"))

    # Create an empty list to store individual DataFrames
    dfs = []

    # Read each CSV file and append it to the list
    for file in csv_files:
        df = pd.read_csv(file, index_col=[0])
        dfs.append(df)

    # Concatenate all DataFrames in the list
    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df


if __name__ == "__main__":
    # Load emulator and metadata
    data_dict, _ = assemble_site_reference_data()
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

    # Usage example
    directory_path = os.path.expanduser(
        "~/EMOD-calibration/calibration/plots/joint/sgd"
    )
    sgd_df = concatenate_csv_files(directory_path)

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

    likelihood_rows = []
    site_dict = dict(zip(site_tokens.iloc[0].values, site_tokens.columns))

    with torch.no_grad():
        for i, row in sgd_df.iterrows():
            train_params = [
                "Falciparum_MSP_Variants_optim",
                "Falciparum_Nonspecific_Types_optim",
                "MSP1_Merozoite_Kill_Fraction_optim",
                "Max_Individual_Infections_optim",
                "Falciparum_PfEMP1_Variants_optim",
                "Antigen_Switch_Rate_optim",
                "Nonspecific_Antigenicity_Factor_optim",
            ]
            param_set = row[train_params].values
            params = torch.tensor(
                np.float32(param_set), dtype=torch.float, requires_grad=True
            )
            ll = calculate_joint_ll(
                mlp, params, site_tokens, data_dict, min_values, max_values
            )
            ll_row = row.copy()
            ll_row["ll"] = ll
            likelihood_rows.append(ll_row)
    sgd_df_ll = pd.concat(likelihood_rows, axis=1).T
    sgd_df_ll.to_csv(
        os.path.expanduser("~/EMOD-calibration/calibration/plots/joint/sgd_ll.csv")
    )
