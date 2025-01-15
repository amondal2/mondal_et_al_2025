import calendar

from calibration.helpers import get_garki_reference, assemble_site_reference_data
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

from calibration.config import (
    senegal_age_bins,
    namawala_age_bins,
    site_metadata,
)
from calibration.helpers import get_bf_reference


def generate_parasitemia_gametocytemia_plots(site, sim_df):
    # TODO cleanup
    site_df = sim_df[sim_df.Site == site]
    sim_means = (
        site_df.iloc[:, 9:]
        .mean()
        .values[dims["Parasitemia_1"]["begin_idx"] : dims["Parasitemia_1"]["end_idx"]]
    )
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
    n_row = len(age_labels)
    n_col = len(months_for_site)
    fig, axs = plt.subplots(nrows=n_row, ncols=n_col, figsize=(20, 15))
    # Data are ordered by (month, age)
    # Split data into individual sets - filter to only months for which we have data
    sim_means_by_month = np.array_split(sim_means, 12)
    sim_means_by_month_dict = {}
    for month in months_for_site:
        sim_means_by_month_dict[month] = sim_means_by_month[month - 1]
    for i, month in enumerate(months_for_site):
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
            axs[j, (n_col - 1)].set_ylabel(age_labels[j])
            axs[j, (n_col - 1)].yaxis.set_label_position("right")
            axs[j, i].plot(x, reference_values)
            axs[j, i].plot(x, sim_data_by_age_month[j])

    fig.suptitle(f"Parasitemia: {site}")
    fig_path = (
        f"~/EMOD-calibration/calibration/plots/prev_calibration/{site}/parasitemia.png"
    )
    fig.legend(["Reference", "Simulation (Mean)"], loc="lower left")
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

    sim_means = (
        site_df.iloc[:, 9:]
        .mean()
        .values[
            dims["Gametocytemia_1"]["begin_idx"] : dims["Gametocytemia_1"]["end_idx"]
        ]
    )
    n_row = len(age_labels)
    n_col = len(months_for_site)
    fig, axs = plt.subplots(nrows=n_row, ncols=n_col, figsize=(20, 15))
    # Data are ordered by (month, age)
    # Split data into individual sets - filter to only months for which we have data
    sim_means_by_month = np.array_split(sim_means, 12)
    sim_means_by_month_dict = {}
    for month in months_for_site:
        sim_means_by_month_dict[month] = sim_means_by_month[month - 1]
    for i, month in enumerate(months_for_site):
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
            axs[j, (n_col - 1)].set_ylabel(age_labels[j])
            axs[j, (n_col - 1)].yaxis.set_label_position("right")
            axs[j, i].plot(x, reference_values)
            axs[j, i].plot(x, sim_data_by_age_month[j])

    fig.suptitle(f"Gametocytemia: {site}")
    fig_path = f"~/EMOD-calibration/calibration/plots/prev_calibration/{site}/gametocytemia.png"
    fig.legend(["Reference", "Simulation (Mean)"], loc="lower left")
    fig.supxlabel("per uL")
    fig.supylabel("Fraction of Population")
    plt.setp(axs, ylim=(0, 1))
    plt.savefig(os.path.expanduser(fig_path))
    plt.clf()
    return True


def generate_parasitemia_plots(site, sim_df):
    site_df = sim_df[sim_df.Site == site]
    sim_means = (
        site_df.iloc[:, 9:]
        .mean()
        .values[dims["Parasitemia_2"]["begin_idx"] : dims["Parasitemia_2"]["end_idx"]]
    )
    months_for_site = site_metadata[site]["months"]
    reference = get_garki_reference(metadata=site_metadata[site])["prevalence"]
    x = ["0", "<16", "16-70", "70-400", ">400"]
    age_labels = [
        "<1 year",
        "1-4 years",
        "4-8 years",
        "8-18 years",
        "18-28 years",
        "28-43 years",
        ">43 years",
    ]
    reference_age_labels = [1, 4, 8, 18, 28, 43, 70]
    # Eval emulator
    n_row = len(age_labels)
    n_col = len(months_for_site)
    fig, axs = plt.subplots(nrows=n_row, ncols=n_col, figsize=(20, 15))
    # Data are ordered by (month, age)
    # Split data into individual sets - filter to only months for which we have data
    sim_means_by_month = np.array_split(sim_means, 12)
    sim_means_by_month_dict = {}
    for month in months_for_site:
        sim_means_by_month_dict[month] = sim_means_by_month[month - 1]

    for i, month in enumerate(months_for_site):
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
            axs[j, (n_col - 1)].set_ylabel(age_labels[j])
            axs[j, (n_col - 1)].yaxis.set_label_position("right")
            axs[j, i].plot(x, reference_values)
            axs[j, i].plot(x, sim_data_by_age_month[j])

    fig.suptitle(f"Parasitemia: {site}")
    fig_path = (
        f"~/EMOD-calibration/calibration/plots/prev_calibration/{site}/parasitemia.png"
    )
    fig.legend(["Reference", "Simulation (Mean)"], loc="lower left")
    fig.supxlabel("Asexual Parasites per uL")
    fig.supylabel("Fraction of Population")
    plt.setp(axs, ylim=(0, 1))
    plt.savefig(os.path.expanduser(fig_path))
    plt.clf()
    return True


def generate_incidence_plots(site, sim_df):
    site_df = sim_df[sim_df.Site == site]
    sim_means = (
        site_df.iloc[:, 9:]
        .mean()
        .values[dims["Incidence"]["begin_idx"] : dims["Incidence"]["end_idx"]]
    )
    mle_df_row = pd.DataFrame(
        {
            "Age Bin": senegal_age_bins,
            "Reference": pd.Series(data_dict[f"{site}_unscaled"]),
            "Simulation (Mean)": pd.Series(sim_means),
        }
    )
    ax = sns.lineplot(
        x="Age Bin",
        y="value",
        hue="variable",
        data=pd.melt(mle_df_row, ["Age Bin"]),
    )
    title = f"{site}"
    ax.set(xlabel="Age Bin", ylabel="Incidence", title=title)
    plt.gca().legend().set_title("")
    fig_path = (
        f"~/EMOD-calibration/calibration/plots/prev_calibration/{site}/incidence.png"
    )
    plt.savefig(os.path.expanduser(fig_path))
    plt.clf()
    return True


def generate_prevalence_plots(site, sim_df):
    site_df = sim_df[sim_df.Site == site]
    sim_means = (
        site_df.iloc[:, 9:]
        .mean()
        .values[dims["PfPr"]["begin_idx"] : dims["PfPr"]["end_idx"]]
    )
    mle_df_row = pd.DataFrame(
        {
            "Age Bin": namawala_age_bins,
            "Reference": pd.Series(data_dict["namawala_1991"]),
            "Simulation (Mean)": pd.Series(sim_means),
        }
    )
    ax = sns.lineplot(
        x="Age Bin",
        y="value",
        hue="variable",
        data=pd.melt(mle_df_row, ["Age Bin"]),
    )
    title = f"{site}"
    ax.set(xlabel="Age Bin", ylabel="Prevalence", title=title)
    plt.gca().legend().set_title("")
    fig_path = (
        f"~/EMOD-calibration/calibration/plots/prev_calibration/{site}/prevalence.png"
    )
    plt.savefig(os.path.expanduser(fig_path))
    plt.clf()
    return True


if __name__ == "__main__":
    _, data_dict = assemble_site_reference_data()

    # Load simulation data
    path = os.path.expanduser(
        "~/EMOD-calibration/emulator/scaled_df_prev_calibration.pkl"
    )
    sim_df = pd.read_pickle(path)
    with open(
        os.path.expanduser(
            "~/EMOD-calibration/emulator/output_dimensions_aggregate_prev_calibration.json"
        ),
        "r",
    ) as content:
        dims = json.load(content)

    generate_prevalence_plots("namawala_1991", sim_df)
    for site in ["ndiop_1993", "dielmo_1990"]:
        generate_incidence_plots(site, sim_df)

    for site in ["matsari_1970", "rafin_marke_1970", "sugungum_1970"]:
        generate_parasitemia_plots(site, sim_df)

    for site in ["dapelogo_2007", "laye_2007"]:
        generate_parasitemia_gametocytemia_plots(site, sim_df)
