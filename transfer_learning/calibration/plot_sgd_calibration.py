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
from simulations import params
from transfer_learning.helpers import get_latest_emulator, site_names
from transfer_learning.mlp import MLP
from torch import nn
import torch
import matplotlib.pyplot as plt

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


def generate_prevalence_plots(
    site, loss_df, min_loss_df, mlp, min_values, max_values, exclude_site
):
    site_df = loss_df[loss_df.Site == site_tokens[site].iloc[0]].reset_index(drop=True)
    for idx, row in min_loss_df.reset_index(drop=True).iterrows():
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
        inputs = torch.tensor(
            site_df_row.iloc[0:20].values.astype("float"),
            dtype=torch.float32,
        ).reshape(1, 20)
        inputs = normalize(inputs, min_values, max_values, num_features=19)
        outputs = mlp(inputs)
        emulator_outputs_pfpr = outputs["PfPr"]
        sim_means = site_df_row[site_df_row.index.str.contains("mean")][
            dims["PfPr"]["begin_idx"] : dims["PfPr"]["end_idx"]
        ].values.flatten()
        mle_df_row = pd.DataFrame(
            {
                "Age Bin": namawala_age_bins,
                "Emulator": pd.Series(emulator_outputs_pfpr.tolist()[0]),
                "Simulation (Mean)": pd.Series(sim_means),
            }
        )
        ax = sns.lineplot(
            x="Age Bin",
            y="value",
            hue="variable",
            data=pd.melt(mle_df_row, ["Age Bin"]),
            palette=["#009E73", "#CC79A7"],
            linewidth=2,
        )
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
        plt.setp(ax, ylim=(0.3, 1.0))
        ax.get_legend().remove()
        fig_path = f"~/EMOD-calibration/transfer_learning/calibration/plots/exclude_{exclude_site}/joint/sgd/{site}/prevalence_{idx}.pdf"
        plt.savefig(os.path.expanduser(fig_path))
        plt.clf()
    return True


def generate_incidence_plots(
    site, loss_df, min_loss_df, mlp, min_values, max_values, exclude_site
):
    site_df = loss_df[loss_df.Site == site_tokens[site].iloc[0]].reset_index(drop=True)

    # Eval emulator
    for idx, row in min_loss_df.reset_index(drop=True).iterrows():
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
        inputs = torch.tensor(
            site_df_row.iloc[0:20].values.astype("float"),
            dtype=torch.float32,
        ).reshape(1, 20)
        inputs = normalize(inputs, min_values, max_values, num_features=19)
        outputs = mlp(inputs)
        emulator_outputs_incidence = outputs["Incidence"]
        sim_means = site_df_row[site_df_row.index.str.contains("mean")][
            dims["Incidence"]["begin_idx"] : dims["Incidence"]["end_idx"]
        ].values.flatten()

        # Rescale to original values
        unscaled_key = f"{site}_unscaled"
        unscaled_data = data_dict[unscaled_key]
        max_val = np.max(unscaled_data)
        min_val = np.min(unscaled_data)
        emulator_scaled = (
            pd.Series(emulator_outputs_incidence.tolist()[0]) * (max_val - min_val)
        ) + min_val
        sim_scaled = (pd.Series(sim_means) * (max_val - min_val)) + min_val

        mle_df_row = pd.DataFrame(
            {
                "Age Bin": senegal_age_bins,
                "Emulator": emulator_scaled,
                "Simulation (Mean)": sim_scaled,
            }
        )
        ax = sns.lineplot(
            x="Age Bin",
            y="value",
            hue="variable",
            data=pd.melt(mle_df_row, ["Age Bin"]),
            palette=["#009E73", "#CC79A7"],
            linewidth=2,
        )
        ax.grid(True)
        ax.margins(x=0, y=0)
        trials = np.array(site_metadata[site]["population_by_age_bin"])
        observations = trials * np.array(site_metadata[site]["Incidence"])
        errs = (np.sqrt(observations) / trials).tolist()
        ax.errorbar(
            x=senegal_age_bins,
            y=pd.Series(unscaled_data),
            yerr=errs,
            color="#E69F00",
            elinewidth=1.5,
            linewidth=2,
            marker="o",
            clip_on=False,
            zorder=10,
        )
        title = f"{site_names[site]}"
        ax.set(xlabel="Age (years)", ylabel="Annual Clinical Incidence")
        ax.set_title(title, fontweight="bold")
        ax.get_legend().remove()
        if site == "ndiop_1993":
            max_val = 3.5
        else:
            max_val = 7.0
        plt.setp(ax, ylim=(0, max_val))
        fig_path = f"~/EMOD-calibration/transfer_learning/calibration/plots/exclude_{exclude_site}/joint/sgd/{site}/incidence_{idx}.pdf"
        plt.savefig(os.path.expanduser(fig_path))
        plt.clf()
    return True


def generate_parasitemia_gametocytemia_plots(
    site, loss_df, min_loss_df, mlp, min_values, max_values, exclude_site
):
    # TODO cleanup
    site_df = loss_df[loss_df.Site == site_tokens[site].iloc[0]].reset_index(drop=True)
    months_for_site = site_metadata[site]["months"]
    reference = get_bf_reference(metadata=site_metadata[site])["prevalence"]
    reference = reference.loc[
        reference.index.get_level_values("Channel")
        == "Smeared PfPR by Parasitemia and Age Bin"
    ]
    x = ["0", "50", "500", "5000", "50000", "inf"]
    age_labels = [
        "<5 year",
        "5-15 years",
        ">15 years",
    ]
    reference_age_labels = [5, 15, 80]
    for idx, row in min_loss_df.reset_index(drop=True).iterrows():
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
        inputs = torch.tensor(
            site_df_row.iloc[0:20].values.astype("float"), dtype=torch.float32
        ).reshape(1, 20)
        inputs = normalize(inputs, min_values, max_values, num_features=19)
        outputs = mlp(inputs)
        emulator_outputs_parasitemia = outputs["Parasitemia_1"][0].detach()
        sim_means = site_df_row[site_df_row.index.str.contains("mean")][
            dims["Parasitemia_1"]["begin_idx"] : dims["Parasitemia_1"]["end_idx"]
        ].values.flatten()
        n_row = len(age_labels)
        n_col = len(months_for_site)
        sns.set_style("darkgrid")
        fig, axs = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=(5, 5), constrained_layout=True
        )
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"

        # Data are ordered by (month, age)
        # Split data into individual sets - filter to only months for which we have data
        emulator_outputs_by_month = np.array_split(emulator_outputs_parasitemia, 12)
        emulator_means_by_month_dict = {}
        for month in months_for_site:
            emulator_means_by_month_dict[month] = emulator_outputs_by_month[month - 1]

        sim_means_by_month = np.array_split(sim_means, 12)
        sim_means_by_month_dict = {}
        for month in months_for_site:
            sim_means_by_month_dict[month] = sim_means_by_month[month - 1]

        for i, month in enumerate(months_for_site):
            emulator_data_for_month = emulator_means_by_month_dict[month]
            emulator_data_by_age_month = np.array_split(
                emulator_data_for_month, dims["Parasitemia_1"]["n_age"]
            )
            sim_data_for_month = sim_means_by_month_dict[month]
            sim_data_by_age_month = np.array_split(
                sim_data_for_month, dims["Parasitemia_1"]["n_age"]
            )
            curr_month = month
            axs[0, i].set_title(calendar.month_name[curr_month])
            for j in range(n_row):
                curr_age = reference_age_labels[j]
                reference_values = reference.loc[
                    (reference.index.get_level_values("Date") == curr_month)
                    & (reference.index.get_level_values("Age Bin") == curr_age)
                ]["Counts"].values
                trials = reference.loc[
                    (reference.index.get_level_values("Date") == curr_month)
                    & (reference.index.get_level_values("Age Bin") == curr_age)
                ]["Counts_tot"].values
                errs = (np.sqrt(reference_values / trials)).tolist()
                errs = np.abs(np.array(errs).T)
                axs[j, i].margins(x=0, y=0)
                axs[j, i].errorbar(
                    x=x,
                    y=reference_values,
                    yerr=errs,
                    color="#E69F00",
                    elinewidth=1.5,
                    linewidth=2,
                    marker="o",
                    clip_on=False,
                    zorder=10,
                )
                sns.lineplot(
                    x=x,
                    y=emulator_data_by_age_month[j],
                    label="Emulator",
                    color="#009E73",
                    ax=axs[j, i],
                    legend=False,
                    lw=2,
                )
                sns.lineplot(
                    x=x,
                    y=sim_data_by_age_month[j],
                    label="Simulation (Mean)",
                    color="#CC79A7",
                    ax=axs[j, i],
                    legend=False,
                    linewidth=2,
                )
                axs[j, i].tick_params("x", labelrotation=45)
                axs[j, i].set_ylabel(None)
                axs[j, (n_col - 1)].set_ylabel(
                    age_labels[j], rotation="horizontal", labelpad=35
                )
                axs[j, (n_col - 1)].yaxis.set_label_position("right")
                if j != n_row - 1:
                    axs[j, i].set_xticklabels("")
                if i > 0:
                    axs[j, i].set_yticklabels("")

        fig.suptitle(f"Parasitemia", fontweight="bold")
        fig_path = f"~/EMOD-calibration/transfer_learning/calibration/plots/exclude_{exclude_site}/joint/sgd/{site}/parasitemia_{idx}.pdf"
        fig.supxlabel("per uL")
        fig.supylabel("Fraction of Population")
        plt.setp(axs, ylim=(0, 1))
        plt.savefig(os.path.expanduser(fig_path))
        plt.clf()

    reference = get_bf_reference(metadata=site_metadata[site])["prevalence"]
    reference = reference.loc[
        reference.index.get_level_values("Channel")
        == "Smeared PfPR by Gametocytemia and Age Bin"
    ]

    for idx, row in min_loss_df.reset_index(drop=True).iterrows():
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
        inputs = torch.tensor(
            site_df_row.iloc[0:20].values.astype("float"), dtype=torch.float32
        ).reshape(1, 20)
        inputs = normalize(inputs, min_values, max_values, num_features=19)
        outputs = mlp(inputs)
        emulator_outputs_gametocytemia = outputs["Gametocytemia_1"][0].detach()
        sim_means = site_df_row[site_df_row.index.str.contains("mean")][
            dims["Gametocytemia_1"]["begin_idx"] : dims["Gametocytemia_1"]["end_idx"]
        ].values.flatten()
        n_row = len(age_labels)
        n_col = len(months_for_site)
        sns.set_style("darkgrid")
        fig, axs = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=(5, 5), constrained_layout=True
        )
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"

        # Data are ordered by (month, age)
        # Split data into individual sets - filter to only months for which we have data
        emulator_outputs_by_month = np.array_split(emulator_outputs_gametocytemia, 12)
        emulator_means_by_month_dict = {}
        for month in months_for_site:
            emulator_means_by_month_dict[month] = emulator_outputs_by_month[month - 1]

        sim_means_by_month = np.array_split(sim_means, 12)
        sim_means_by_month_dict = {}
        for month in months_for_site:
            sim_means_by_month_dict[month] = sim_means_by_month[month - 1]

        for i, month in enumerate(months_for_site):
            emulator_data_for_month = emulator_means_by_month_dict[month]
            emulator_data_by_age_month = np.array_split(
                emulator_data_for_month, dims["Gametocytemia_1"]["n_age"]
            )
            sim_data_for_month = sim_means_by_month_dict[month]
            sim_data_by_age_month = np.array_split(
                sim_data_for_month, dims["Gametocytemia_1"]["n_age"]
            )
            curr_month = month
            axs[0, i].set_title(calendar.month_name[curr_month])
            for j in range(n_row):
                curr_age = reference_age_labels[j]
                reference_values = reference.loc[
                    (reference.index.get_level_values("Date") == curr_month)
                    & (reference.index.get_level_values("Age Bin") == curr_age)
                ]["Counts"].values
                trials = reference.loc[
                    (reference.index.get_level_values("Date") == curr_month)
                    & (reference.index.get_level_values("Age Bin") == curr_age)
                ]["Counts_tot"].values
                errs = (np.sqrt(reference_values / trials)).tolist()
                errs = np.abs(np.array(errs).T)
                axs[j, i].margins(x=0, y=0)
                axs[j, i].errorbar(
                    x=x,
                    y=reference_values,
                    yerr=errs,
                    color="#E69F00",
                    elinewidth=1.5,
                    linewidth=2,
                    marker="o",
                    clip_on=False,
                    zorder=10,
                )
                sns.lineplot(
                    x=x,
                    y=emulator_data_by_age_month[j],
                    label="Emulator",
                    color="#009E73",
                    ax=axs[j, i],
                    legend=False,
                    lw=2,
                )
                sns.lineplot(
                    x=x,
                    y=sim_data_by_age_month[j],
                    label="Simulation (Mean)",
                    color="#CC79A7",
                    ax=axs[j, i],
                    legend=False,
                    lw=2,
                )
                axs[j, i].tick_params("x", labelrotation=45)
                axs[j, i].set_ylabel(None)
                axs[j, (n_col - 1)].set_ylabel(
                    age_labels[j], rotation="horizontal", labelpad=35
                )
                axs[j, (n_col - 1)].yaxis.set_label_position("right")
                if j != n_row - 1:
                    axs[j, i].set_xticklabels("")
                if i > 0:
                    axs[j, i].set_yticklabels("")

        fig.suptitle(f"Gametocytemia", fontweight="bold")
        fig_path = f"~/EMOD-calibration/transfer_learning/calibration/plots/exclude_{exclude_site}/joint/sgd/{site}/gametocytemia_{idx}.pdf"
        fig.supxlabel("per uL")
        fig.supylabel("Fraction of Population")
        plt.setp(axs, ylim=(0, 1))
        plt.savefig(os.path.expanduser(fig_path))
        plt.clf()
    return True


def generate_parasitemia_plots(
    site, loss_df, min_loss_df, mlp, min_values, max_values, exclude_site
):
    site_df = loss_df[loss_df.Site == site_tokens[site].iloc[0]].reset_index(drop=True)
    months_for_site = site_metadata[site]["months"]
    reference = get_garki_reference(metadata=site_metadata[site])["prevalence"]
    x = ["0", "<16", "16-70", "70-400", ">400"]
    age_labels = [
        "<1 year",
        "0-4 years",
        "4-8 years",
        "8-18 years",
        "18-28 years",
        "28-43 years",
        ">43 years",
    ]
    reference_age_labels = [1, 4, 8, 18, 28, 43, 70]
    # Eval emulator
    for idx, row in min_loss_df.reset_index(drop=True).iterrows():
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
        inputs = torch.tensor(
            site_df_row.iloc[0:20].values.astype("float"), dtype=torch.float32
        ).reshape(1, 20)
        inputs = normalize(inputs, min_values, max_values, num_features=19)
        outputs = mlp(inputs)
        emulator_outputs_parasitemia = outputs["Parasitemia_2"][0].detach()
        sim_means = site_df_row[site_df_row.index.str.contains("mean")][
            dims["Parasitemia_2"]["begin_idx"] : dims["Parasitemia_2"]["end_idx"]
        ].values.flatten()
        n_row = len(age_labels)
        n_col = len(months_for_site)
        sns.set_style("darkgrid")
        fig, axs = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=(15, 10), constrained_layout=True
        )
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"

        # Data are ordered by (month, age)
        # Split data into individual sets - filter to only months for which we have data
        emulator_outputs_by_month = np.array_split(emulator_outputs_parasitemia, 12)
        emulator_means_by_month_dict = {}
        for month in months_for_site:
            emulator_means_by_month_dict[month] = emulator_outputs_by_month[month - 1]

        sim_means_by_month = np.array_split(sim_means, 12)
        sim_means_by_month_dict = {}
        for month in months_for_site:
            sim_means_by_month_dict[month] = sim_means_by_month[month - 1]

        for i, month in enumerate(months_for_site):
            emulator_data_for_month = emulator_means_by_month_dict[month]
            emulator_data_by_age_month = np.array_split(
                emulator_data_for_month, dims["Parasitemia_2"]["n_age"]
            )
            sim_data_for_month = sim_means_by_month_dict[month]
            sim_data_by_age_month = np.array_split(
                sim_data_for_month, dims["Parasitemia_2"]["n_age"]
            )
            curr_month = month
            axs[0, i].set_title(calendar.month_name[curr_month])

            for j in range(n_row):
                curr_age = reference_age_labels[j]
                reference_values = reference.loc[
                    (reference.index.get_level_values("Date") == curr_month)
                    & (reference.index.get_level_values("Age Bin") == curr_age)
                ]["Counts"].values
                trials = reference.loc[
                    (reference.index.get_level_values("Date") == curr_month)
                    & (reference.index.get_level_values("Age Bin") == curr_age)
                ]["Counts_tot"].values
                errs = (np.sqrt(reference_values / trials)).tolist()
                errs = np.abs(np.array(errs).T)
                axs[j, i].margins(x=0, y=0)
                axs[j, i].errorbar(
                    x=x,
                    y=reference_values,
                    yerr=errs,
                    color="#E69F00",
                    elinewidth=1.5,
                    linewidth=2,
                    marker="o",
                    clip_on=False,
                    zorder=10,
                )
                sns.lineplot(
                    x=x,
                    y=emulator_data_by_age_month[j],
                    label="Emulator",
                    color="#009E73",
                    ax=axs[j, i],
                    legend=False,
                    lw=2,
                )
                sns.lineplot(
                    x=x,
                    y=sim_data_by_age_month[j],
                    label="Simulation (Mean)",
                    color="#CC79A7",
                    ax=axs[j, i],
                    legend=False,
                    lw=2,
                )
                axs[j, i].tick_params("x", labelrotation=45)
                axs[j, i].set_ylabel(None)
                axs[j, (n_col - 1)].set_ylabel(
                    age_labels[j], rotation="horizontal", labelpad=40
                )
                axs[j, (n_col - 1)].yaxis.set_label_position("right")
                if j != n_row - 1:
                    axs[j, i].set_xticklabels("")
                if i > 0:
                    axs[j, i].set_yticklabels("")

        # fig.suptitle(f"Parasitemia: {site_names[site]}")
        fig_path = f"~/EMOD-calibration/transfer_learning/calibration/plots/exclude_{exclude_site}/joint/sgd/{site}/parasitemia_{idx}.pdf"
        fig.supxlabel("Asexual Parasites per uL")
        fig.supylabel("Fraction of Population")
        plt.setp(axs, ylim=(0, 1))
        plt.savefig(os.path.expanduser(fig_path))
        plt.clf()
    return True


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
    return joint_loss_df.sort_values(by=["joint_sim_loss"], ascending=True).head(10)


if __name__ == "__main__":
    # Load emulator and metadata
    for exclude_site in [params.sites[7]]:
        _, data_dict = assemble_site_reference_data()
        device = torch.device("cpu")
        model_path, config_path = get_latest_emulator(exclude_site=exclude_site)
        with open(os.path.expanduser(config_path)) as fp:
            config = json.load(fp)

        site_tokens = pd.read_csv(
            f"~/EMOD-calibration/transfer_learning/simulations/exclude_{exclude_site}/site_tokens.csv",
            index_col=0,
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

        # Load simulation data
        path = os.path.expanduser(
            f"~/EMOD-calibration/transfer_learning/simulations/exclude_{exclude_site}/scaled_df.pkl"
        )
        sim_df = pd.read_pickle(path)

        # Normalization constants
        min_values = pd.read_csv(
            os.path.expanduser(
                f"~/EMOD-calibration/transfer_learning/simulations/exclude_{exclude_site}/param_mins_transfer.csv"
            ),
            index_col=0,
        )
        min_values = torch.tensor(
            min_values.T.values, dtype=torch.float32, device=device
        )
        max_values = pd.read_csv(
            os.path.expanduser(
                f"~/EMOD-calibration/transfer_learning/simulations/exclude_{exclude_site}/param_max_transfer.csv"
            ),
            index_col=0,
        )
        max_values = torch.tensor(
            max_values.T.values, dtype=torch.float32, device=device
        )

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

        sim_rows = []
        site_dict = dict(zip(site_tokens.iloc[0].values, site_tokens.columns))

        with torch.no_grad():
            for df in dfs:
                sim_means = df.iloc[:, 22:].mean().rename("mean_{}".format)
                site = site_dict[df.Site.iloc[0]]

                # Eval emulator
                input_params = df.iloc[:, :20].iloc[0]
                sim_loss = calculate_sim_loss_for_site(
                    sim_means=sim_means,
                    reference_data=data_dict[site],
                    site=site,
                    dims=dims,
                )

                row = pd.concat(
                    [
                        input_params,
                        pd.Series({"sim_loss": sim_loss, "Site": df.Site.iloc[0]}),
                        sim_means,
                    ]
                )
                sim_rows.append(row)
        loss_df = pd.concat(sim_rows, axis=1).transpose()
        min_loss_df = find_min_joint_loss(loss_df)
        data_dir = os.path.expanduser(
            f"~/EMOD-calibration/transfer_learning/calibration/data/exclude_{exclude_site}/"
        )
        for filename in os.listdir(data_dir):
            if filename.startswith("sgd_loss_") and filename.endswith(".csv"):
                file_path = os.path.join(data_dir, filename)

                # Read the CSV file and append to the list
                df = pd.read_csv(file_path)
                dfs.append(df)
        emulator_loss_df = pd.concat(dfs, ignore_index=True)
        min_loss_df = (
            pd.merge(min_loss_df, emulator_loss_df)
            .sort_values(by=["loss"], ascending=True)
            .head(10)
        )

        generate_prevalence_plots(
            "namawala_1991",
            loss_df,
            min_loss_df,
            mlp,
            min_values,
            max_values,
            exclude_site,
        )

        for site in ["ndiop_1993", "dielmo_1990"]:
            generate_incidence_plots(
                site, loss_df, min_loss_df, mlp, min_values, max_values, exclude_site
            )

        for site in ["matsari_1970", "rafin_marke_1970", "sugungum_1970"]:
            generate_parasitemia_plots(
                site, loss_df, min_loss_df, mlp, min_values, max_values, exclude_site
            )
        for site in ["dapelogo_2007", "laye_2007"]:
            generate_parasitemia_gametocytemia_plots(
                site, loss_df, min_loss_df, mlp, min_values, max_values, exclude_site
            )
