"""
Calibration to reference data using nearest neighbors-based approach.
Calibrate across sites.

"""
import itertools

import pandas as pd
import os
import json
import numpy as np
import seaborn as sns
import calendar
from scipy.stats import binom

from calibration.calibrate_likelihood_joint import (
    calculate_sim_likelihood_for_site,
)
from calibration.config import (
    site_metadata,
    namawala_age_bins,
    senegal_age_bins,
)
from calibration.helpers import (
    beta_binomial_likelihood,
    gamma_poisson_likelihood,
    dirichlet_multinomial_likelihood,
    calculate_observations,
    get_garki_reference,
    assemble_site_reference_data,
    get_bf_reference,
)
from emulator.helpers import get_latest_emulator, site_names
from emulator.multisite.mlp import MLP
from torch import nn
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("ggplot")
plt.rcParams["pdf.fonttype"] = 42


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


def format_sim_data_garki(sim_outputs=None, months=None, dims=None):
    return generate_sim_df(
        dims=dims,
        output_key="Parasitemia_2",
        channel_name="Smeared PfPR by Parasitemia and Age Bin",
        sim_outputs=sim_outputs,
        months=months,
    )


def generate_sim_df(dims, output_key, channel_name, sim_outputs, months):
    n_month = 12
    n_age = dims[output_key]["n_age"]
    n_density = dims[output_key]["n_density"]
    combinations = list(
        itertools.product(
            range(1, n_month + 1), range(1, n_age + 1), range(0, n_density)
        )
    )
    sim_df = pd.DataFrame(
        combinations, columns=["Date", "Age Bin", "PfPR Bin"]
    ).sort_values(by=["Date", "Age Bin"])
    sim_df["Counts"] = sim_outputs[
        dims[output_key]["begin_idx"] : dims[output_key]["end_idx"]
    ].values.flatten()
    sim_df["Channel"] = channel_name
    sim_df = sim_df.set_index(["Channel", "Date", "Age Bin", "PfPR Bin"])
    sim_df = sim_df.loc[sim_df.index.get_level_values("Date").isin(months)]
    return sim_df


def calculate_sim_loss_prevalence(
    sim_outputs=None, reference_data=None, site=None, dims=None
):
    # Format dataframe into Observations / Trials format for LL calculations
    sim_outputs = sim_outputs[
        dims["PfPr"]["begin_idx"] : dims["PfPr"]["end_idx"]
    ].values.flatten()
    sim_outputs = torch.tensor(sim_outputs).unsqueeze(0)
    reference_data = torch.tensor(reference_data).unsqueeze(0)
    loss_fn = nn.MSELoss()
    return loss_fn(sim_outputs, reference_data)


def calculate_sim_loss_incidence(
    sim_outputs=None, reference_data=None, site=None, dims=None
):
    # Format dataframe into Observations / Trials format for LL calculations
    sim_outputs = sim_outputs[
        dims["Incidence"]["begin_idx"] : dims["Incidence"]["end_idx"]
    ].values.flatten()
    sim_outputs = torch.tensor(sim_outputs).unsqueeze(0)
    reference_data = torch.tensor(reference_data).unsqueeze(0)
    loss_fn = nn.MSELoss()
    return loss_fn(sim_outputs, reference_data)


def calculate_sim_loss_parasitemia(
    sim_outputs=None, reference_data=None, site=None, dims=None
):
    sim_df = format_sim_data_garki(
        sim_outputs, months=site_metadata[site]["months"], dims=dims
    ).sort_values(by=["Age Bin", "Date"])
    reference_df = reference_data.sort_values(by=["Age Bin", "Date"])
    loss_fn = nn.MSELoss()
    emulator_values = torch.tensor(sim_df["Counts"].values).unsqueeze(0)
    reference_values = torch.tensor(reference_df["Counts"].values).unsqueeze(0)
    return loss_fn(emulator_values, reference_values)


def format_sim_data_burkinafaso(sim_outputs=None, months=None, dims=None):
    # Data are ordered by (month, age)
    # Create data frame and rebin/index like reference data
    parasitemia_df = generate_sim_df(
        dims=dims,
        output_key="Parasitemia_1",
        channel_name="Smeared PfPR by Parasitemia and Age Bin",
        sim_outputs=sim_outputs,
        months=months,
    )

    gametocytemia_df = generate_sim_df(
        dims=dims,
        output_key="Gametocytemia_1",
        channel_name="Smeared PfPR by Gametocytemia and Age Bin",
        sim_outputs=sim_outputs,
        months=months,
    )

    return pd.concat([gametocytemia_df, parasitemia_df])


def calculate_sim_loss_parasitemia_gametocytemia(
    sim_outputs=None, reference_data=None, site=None, dims=None
):
    sim_df = format_sim_data_burkinafaso(
        sim_outputs, months=site_metadata[site]["months"], dims=dims
    ).sort_values(by=["Age Bin", "Date"])
    reference_df = reference_data.sort_values(by=["Age Bin", "Date"]).sort_values(
        by=["Age Bin", "Date"]
    )
    loss_fn = nn.MSELoss()
    emulator_values = torch.tensor(sim_df["Counts"].values).unsqueeze(0)
    reference_values = torch.tensor(reference_df["Counts"].values).unsqueeze(0)
    return loss_fn(emulator_values, reference_values)


def calculate_sim_loss_for_site(sim_means, reference_data, site, dims):
    loss_functions = {
        "namawala_1991": calculate_sim_loss_prevalence,
        "ndiop_1993": calculate_sim_loss_incidence,
        "dielmo_1990": calculate_sim_loss_incidence,
        "matsari_1970": calculate_sim_loss_parasitemia,
        "rafin_marke_1970": calculate_sim_loss_parasitemia,
        "sugungum_1970": calculate_sim_loss_parasitemia,
        "dapelogo_2007": calculate_sim_loss_parasitemia_gametocytemia,
        "laye_2007": calculate_sim_loss_parasitemia_gametocytemia,
    }
    lf = loss_functions[site]
    return lf(
        sim_outputs=sim_means,
        reference_data=reference_data,
        site=site,
        dims=dims,
    )


def find_min_joint_loss(loss_df):
    train_params = [
        "Falciparum_MSP_Variants",
        "Falciparum_Nonspecific_Types",
        "MSP1_Merozoite_Kill_Fraction",
        "Max_Individual_Infections",
        "Falciparum_PfEMP1_Variants",
        "Antigen_Switch_Rate",
        "Nonspecific_Antigenicity_Factor",
    ]
    dfs = [y for x, y in loss_df.groupby(train_params, as_index=False)]
    joint_loss_rows = []
    for df in dfs:
        input_params = df.iloc[:, :7].iloc[0]
        sim_loss = np.sum(df["sim_loss"])
        row = pd.concat(
            [
                input_params,
                pd.Series({"joint_sim_loss": sim_loss}),
            ]
        )
        joint_loss_rows.append(row)
    joint_loss_df = pd.concat(joint_loss_rows, axis=1).transpose()
    # Bottom 10 loss values
    return joint_loss_df.sort_values(by=["joint_sim_loss"], ascending=True)


def find_max_joint_likelihood(likelihood_df):
    train_params = [
        "Falciparum_MSP_Variants",
        "Falciparum_Nonspecific_Types",
        "MSP1_Merozoite_Kill_Fraction",
        "Max_Individual_Infections",
        "Falciparum_PfEMP1_Variants",
        "Antigen_Switch_Rate",
        "Nonspecific_Antigenicity_Factor",
    ]
    dfs = [y for x, y in likelihood_df.groupby(train_params, as_index=False)]
    joint_likelihood_rows = []
    for df in dfs:
        input_params = df.iloc[:, :7].iloc[0]
        sim_likelihood = np.sum(df["sim_likelihood"])
        row = pd.concat(
            [
                input_params,
                pd.Series({"sim_likelihood": sim_likelihood}),
            ]
        )
        joint_likelihood_rows.append(row)
    joint_likelihood_df = pd.concat(joint_likelihood_rows, axis=1).transpose()
    # Top 10 max likelihoods
    return joint_likelihood_df.sort_values(by=["sim_likelihood"], ascending=False)


if __name__ == "__main__":
    # Load emulator and metadata

    _, data_dict = assemble_site_reference_data()
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

    # Load calibration data data
    path = os.path.expanduser("~/EMOD-calibration/calibration/plots/joint/sgd_ll.csv")
    sgd_df = pd.read_csv(path)

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

    # Generate trace df
    train_params = [
        "Falciparum_MSP_Variants_optim",
        "Falciparum_Nonspecific_Types_optim",
        "MSP1_Merozoite_Kill_Fraction_optim",
        "Max_Individual_Infections_optim",
        "Falciparum_PfEMP1_Variants_optim",
        "Antigen_Switch_Rate_optim",
        "Nonspecific_Antigenicity_Factor_optim",
    ]
    sgd_df_ll = sgd_df.sort_values(by=["ll"], ascending=False).head(25)
    sgd_df_loss = sgd_df.sort_values(by=["loss"], ascending=True).head(25)

    # Generate plots for namawala
    site = "namawala_1991"
    token = site_tokens[site].iloc[0]
    all_dfs = []
    for idx, row in sgd_df_ll.reset_index(drop=True).iterrows():
        params = torch.tensor(
            np.float32(row[train_params].values), dtype=torch.float, requires_grad=True
        )
        inputs = torch.cat((params, torch.tensor([token]))).unsqueeze(0)
        inputs = normalize(inputs, min_values, max_values, num_features=7)
        outputs = mlp(inputs)["PfPr"]
        df_row = pd.DataFrame(
            {
                "age_bin": namawala_age_bins,
                "pfpr": outputs.tolist()[0],
                "idx": idx,
                "ll": row["ll"],
            }
        )
        all_dfs.append(df_row)

    sgd_df_formatted = pd.concat(all_dfs, axis=0)
    ax = sns.lineplot(
        data=sgd_df_formatted,
        x="age_bin",
        y="pfpr",
        hue="ll",
        # marker="o",
        lw=1,
        alpha=0.75,
        palette=sns.light_palette("#009E73", as_cmap=True, reverse=False),
    )
    norm = mpl.colors.Normalize(
        vmin=sgd_df_formatted["ll"].min(), vmax=sgd_df_formatted["ll"].max()
    )
    sm = plt.cm.ScalarMappable(
        cmap=sns.light_palette("#009E73", as_cmap=True, reverse=False), norm=norm
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Goodness-of-fit (log-likelihood)")
    cbar.set_ticks([])

    ax.grid(True)
    ax.margins(x=0, y=0)
    trials = np.array(site_metadata[site]["population_by_age_bin"])
    observations = trials * np.array(site_metadata[site]["PfPr"])
    errs = [
        binom.interval(0.68, n, p=k / n, loc=-k) / n
        for n, k in zip(trials, observations)
    ]
    errs = np.abs(np.array(errs).T)
    ax.errorbar(
        x=namawala_age_bins,
        y=pd.Series(data_dict["namawala_1991"]),
        yerr=errs,
        color="#E69F00",
        elinewidth=1.5,
        linewidth=2,
        marker="o",
        clip_on=False,
        zorder=10,
    )
    title = f"{site_names[site]}"
    ax.set(xlabel="Age (years)", ylabel="Prevalence")
    ax.set_title(title, fontweight="bold")

    plt.setp(ax, ylim=(0.0, 1.0))
    ax.get_legend().remove()
    fig_path = (
        f"~/EMOD-calibration/calibration/plots/joint/sgd/sgd_traces_namawala_ll.pdf"
    )
    plt.savefig(os.path.expanduser(fig_path))
    plt.clf()

    all_dfs = []
    for idx, row in sgd_df_loss.iterrows():
        params = torch.tensor(
            np.float32(row[train_params].values), dtype=torch.float, requires_grad=True
        )
        inputs = torch.cat((params, torch.tensor([token]))).unsqueeze(0)
        inputs = normalize(inputs, min_values, max_values, num_features=7)
        outputs = mlp(inputs)["PfPr"]
        df_row = pd.DataFrame(
            {
                "age_bin": namawala_age_bins,
                "pfpr": outputs.tolist()[0],
                "idx": idx,
                "loss": row["loss"],
            }
        )
        all_dfs.append(df_row)

    sgd_df_formatted = pd.concat(all_dfs, axis=0)
    ax = sns.lineplot(
        data=sgd_df_formatted,
        x="age_bin",
        y="pfpr",
        hue="loss",
        # marker="o",
        palette=sns.light_palette("#009E73", as_cmap=True, reverse=True),
        lw=1,
        alpha=0.75,
    )
    norm = mpl.colors.Normalize(
        vmin=sgd_df_formatted["loss"].min(), vmax=sgd_df_formatted["loss"].max()
    )
    sm = plt.cm.ScalarMappable(
        cmap=sns.light_palette("#009E73", as_cmap=True, reverse=False), norm=norm
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Goodness-of-fit (L2 loss)")
    cbar.set_ticks([])
    ax.grid(True)
    ax.margins(x=0, y=0)
    trials = np.array(site_metadata[site]["population_by_age_bin"])
    observations = trials * np.array(site_metadata[site]["PfPr"])
    errs = [
        binom.interval(0.68, n, p=k / n, loc=-k) / n
        for n, k in zip(trials, observations)
    ]
    errs = np.abs(np.array(errs).T)
    ax.errorbar(
        x=namawala_age_bins,
        y=pd.Series(data_dict["namawala_1991"]),
        yerr=errs,
        color="#E69F00",
        elinewidth=1.5,
        linewidth=2,
        marker="o",
        clip_on=False,
        zorder=10,
    )
    title = f"{site_names[site]}"
    ax.set(xlabel="Age (years)", ylabel="Prevalence")
    ax.set_title(title, fontweight="bold")
    plt.setp(ax, ylim=(0.0, 1.0))
    # Create a colorbar
    ax.get_legend().remove()
    fig_path = (
        f"~/EMOD-calibration/calibration/plots/joint/sgd/sgd_traces_namawala_loss.pdf"
    )
    plt.savefig(os.path.expanduser(fig_path))
    plt.clf()

    sim_df_l2 = pd.read_pickle(
        os.path.expanduser(
            "~/EMOD-calibration/emulator/scaled_df_sgd_l2_calibration.pkl"
        )
    )
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
    dfs = [y for x, y in sim_df_l2.groupby(train_params, as_index=False)]

    sim_rows = []
    site_dict = dict(zip(site_tokens.iloc[0].values, site_tokens.columns))
    with torch.no_grad():
        for df in dfs:
            sim_means = df.iloc[:, 9:].mean().rename("mean_{}".format)
            site = site_dict[df.Site.iloc[0]]

            # Eval emulator
            input_params = df.iloc[:, :8].iloc[0]
            sim_loss = calculate_sim_loss_for_site(
                sim_means=sim_means,
                reference_data=data_dict[site],
                site=site,
                dims=dims,
            )

            row = pd.concat(
                [
                    input_params,
                    pd.Series({"sim_loss": sim_loss.item()}),
                    sim_means,
                ]
            )
            sim_rows.append(row)
    loss_df = pd.concat(sim_rows, axis=1).transpose()
    min_loss_df = find_min_joint_loss(loss_df)
    site = "namawala_1991"
    all_dfs = []
    for idx, row in min_loss_df.iterrows():
        site_df = loss_df[loss_df.Site == site_tokens[site].iloc[0]].reset_index(
            drop=True
        )
        site_df_row = site_df.loc[
            (site_df["Falciparum_MSP_Variants"] == row["Falciparum_MSP_Variants"])
            & (
                site_df["Falciparum_Nonspecific_Types"]
                == row["Falciparum_Nonspecific_Types"]
            )
            & (
                site_df["MSP1_Merozoite_Kill_Fraction"]
                == row["MSP1_Merozoite_Kill_Fraction"]
            )
            & (site_df["Max_Individual_Infections"] == row["Max_Individual_Infections"])
            & (
                site_df["Falciparum_PfEMP1_Variants"]
                == row["Falciparum_PfEMP1_Variants"]
            )
            & (site_df["Antigen_Switch_Rate"] == row["Antigen_Switch_Rate"])
            & (
                site_df["Nonspecific_Antigenicity_Factor"]
                == row["Nonspecific_Antigenicity_Factor"]
            )
        ].transpose()
        sim_means = site_df_row[site_df_row.index.str.contains("mean")][
            dims["PfPr"]["begin_idx"] : dims["PfPr"]["end_idx"]
        ].values.flatten()
        mle_df_row = pd.DataFrame(
            {
                "age_bin": namawala_age_bins,
                "pfpr": pd.Series(sim_means),
                "idx": idx,
                "loss": row["joint_sim_loss"],
            }
        )
        all_dfs.append(mle_df_row)
    sgd_df_formatted = pd.concat(all_dfs, axis=0)
    ax = sns.lineplot(
        data=sgd_df_formatted,
        x="age_bin",
        y="pfpr",
        hue="loss",
        # marker="o",
        palette=sns.light_palette("#CC79A7", as_cmap=True, reverse=True),
        lw=1,
        alpha=0.75,
    )
    norm = mpl.colors.Normalize(
        vmin=sgd_df_formatted["loss"].min(),
        vmax=sgd_df_formatted["loss"].max(),
    )
    sm = plt.cm.ScalarMappable(
        cmap=sns.light_palette("#CC79A7", as_cmap=True, reverse=False), norm=norm
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Goodness-of-fit (L2 loss)")
    cbar.set_ticks([])
    ax.grid(True)
    ax.margins(x=0, y=0)
    trials = np.array(site_metadata[site]["population_by_age_bin"])
    observations = trials * np.array(site_metadata[site]["PfPr"])
    errs = [
        binom.interval(0.68, n, p=k / n, loc=-k) / n
        for n, k in zip(trials, observations)
    ]
    errs = np.abs(np.array(errs).T)
    ax.errorbar(
        x=namawala_age_bins,
        y=pd.Series(data_dict["namawala_1991"]),
        yerr=errs,
        color="#E69F00",
        elinewidth=1.5,
        linewidth=2,
        marker="o",
        clip_on=False,
        zorder=10,
    )
    title = f"{site_names[site]}"
    ax.set(xlabel="Age (years)", ylabel="Prevalence")
    ax.set_title(title, fontweight="bold")
    plt.setp(ax, ylim=(0.0, 1.0))
    # Create a colorbar
    ax.get_legend().remove()
    fig_path = f"~/EMOD-calibration/calibration/plots/joint/sgd/sgd_sim_traces_namawala_loss.pdf"
    plt.savefig(os.path.expanduser(fig_path))
    plt.clf()

    sim_df_ll = pd.read_pickle(
        os.path.expanduser(
            "~/EMOD-calibration/emulator/scaled_df_sgd_ll_calibration.pkl"
        )
    )
    data_dict, _ = assemble_site_reference_data()
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
    dfs = [y for x, y in sim_df_ll.groupby(train_params, as_index=False)]

    sim_rows = []
    site_dict = dict(zip(site_tokens.iloc[0].values, site_tokens.columns))
    with torch.no_grad():
        for df in dfs:
            sim_means = df.iloc[:, 9:].mean().rename("mean_{}".format)
            site = site_dict[df.Site.iloc[0]]

            # Eval emulator
            input_params = df.iloc[:, :8].iloc[0]
            sim_likelihood = calculate_sim_likelihood_for_site(
                sim_means=sim_means,
                reference_data=data_dict[site],
                site=site,
                dims=dims,
            )

            row = pd.concat(
                [
                    input_params,
                    pd.Series(
                        {
                            "sim_likelihood": sim_likelihood["ll"],
                        }
                    ),
                    sim_means,
                ]
            )
            sim_rows.append(row)
    likelihood_df = pd.concat(sim_rows, axis=1).transpose()
    max_likelihood_df = find_max_joint_likelihood(likelihood_df)
    site = "namawala_1991"
    all_dfs = []
    for idx, row in max_likelihood_df.iterrows():
        site_df = likelihood_df[
            likelihood_df.Site == site_tokens[site].iloc[0]
        ].reset_index(drop=True)
        site_df_row = site_df.loc[
            (site_df["Falciparum_MSP_Variants"] == row["Falciparum_MSP_Variants"])
            & (
                site_df["Falciparum_Nonspecific_Types"]
                == row["Falciparum_Nonspecific_Types"]
            )
            & (
                site_df["MSP1_Merozoite_Kill_Fraction"]
                == row["MSP1_Merozoite_Kill_Fraction"]
            )
            & (site_df["Max_Individual_Infections"] == row["Max_Individual_Infections"])
            & (
                site_df["Falciparum_PfEMP1_Variants"]
                == row["Falciparum_PfEMP1_Variants"]
            )
            & (site_df["Antigen_Switch_Rate"] == row["Antigen_Switch_Rate"])
            & (
                site_df["Nonspecific_Antigenicity_Factor"]
                == row["Nonspecific_Antigenicity_Factor"]
            )
        ].transpose()
        sim_means = site_df_row[site_df_row.index.str.contains("mean")][
            dims["PfPr"]["begin_idx"] : dims["PfPr"]["end_idx"]
        ].values.flatten()
        mle_df_row = pd.DataFrame(
            {
                "age_bin": namawala_age_bins,
                "pfpr": pd.Series(sim_means),
                "idx": idx,
                "ll": row["sim_likelihood"],
            }
        )
        all_dfs.append(mle_df_row)
    sgd_df_formatted = pd.concat(all_dfs, axis=0)
    ax = sns.lineplot(
        data=sgd_df_formatted,
        x="age_bin",
        y="pfpr",
        hue="ll",
        # marker="o",
        palette=sns.light_palette("#CC79A7", as_cmap=True, reverse=False),
        lw=1,
        alpha=0.75,
    )
    norm = mpl.colors.Normalize(
        vmin=sgd_df_formatted["ll"].min(),
        vmax=sgd_df_formatted["ll"].max(),
    )
    sm = plt.cm.ScalarMappable(
        cmap=sns.light_palette("#CC79A7", as_cmap=True, reverse=False), norm=norm
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Goodness-of-fit (Log likelihood)")
    cbar.set_ticks([])
    ax.grid(True)
    ax.margins(x=0, y=0)
    trials = np.array(site_metadata[site]["population_by_age_bin"])
    observations = trials * np.array(site_metadata[site]["PfPr"])
    errs = [
        binom.interval(0.68, n, p=k / n, loc=-k) / n
        for n, k in zip(trials, observations)
    ]
    errs = np.abs(np.array(errs).T)
    ax.errorbar(
        x=namawala_age_bins,
        y=pd.Series(data_dict["namawala_1991"]),
        yerr=errs,
        color="#E69F00",
        elinewidth=1.5,
        linewidth=2,
        marker="o",
        clip_on=False,
        zorder=10,
    )
    title = f"{site_names[site]}"
    ax.set(xlabel="Age (years)", ylabel="Prevalence")
    ax.set_title(title, fontweight="bold")
    plt.setp(ax, ylim=(0.0, 1.0))
    # Create a colorbar
    ax.get_legend().remove()
    fig_path = (
        f"~/EMOD-calibration/calibration/plots/joint/sgd/sgd_sim_traces_namawala_ll.pdf"
    )
    plt.savefig(os.path.expanduser(fig_path))
    plt.clf()
