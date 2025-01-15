import calendar
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from torch import nn
import os
import seaborn as sns

from calibration.config import namawala_age_bins, senegal_age_bins, site_metadata
from emulator.helpers import get_latest_emulator, site_names
from emulator.multisite.mlp import MLP


def generate_prevalence_plots(site, loss_df, mlp):
    site_df = loss_df[loss_df.Site == site_tokens[site].iloc[0]].reset_index(drop=True)

    # Bottom 10 loss values (best fits)
    mle_df = (
        site_df.sort_values(by=["loss"], ascending=True).head(10).reset_index(drop=True)
    )
    # Eval emulator
    for idx, row in mle_df.iterrows():
        inputs = torch.tensor(
            row.iloc[0:8].values.astype("float"), dtype=torch.float32
        ).reshape(1, 8)
        inputs = normalize(inputs, min_values, max_values, num_features=7)
        outputs = mlp(inputs)
        emulator_outputs_pfpr = outputs["PfPr"]
        sim_means = row[row.index.str.contains("mean")][
            dims["PfPr"]["begin_idx"] : dims["PfPr"]["end_idx"]
        ].values
        mle_df_row = pd.DataFrame(
            {
                "Age Bin": namawala_age_bins,
                "Emulator": pd.Series(emulator_outputs_pfpr.tolist()[0]),
                "Simulation (Mean)": pd.Series(sim_means),
            }
        )
        sns.set_style("darkgrid")
        ax = sns.lineplot(
            x="Age Bin",
            y="value",
            hue="variable",
            data=pd.melt(mle_df_row, ["Age Bin"]),
            palette=["#009E73", "#0072B2"],
        )
        title = f"{site_names[site]}"
        ax.set(xlabel="Age (years)", ylabel="Prevalence", title=title)
        plt.setp(ax, ylim=(0, 1.1))
        plt.gca().legend().set_title("")
        fig_path = (
            f"~/EMOD-calibration/emulator/multisite/plots/{site}/prevalence_{idx}.png"
        )
        plt.savefig(os.path.expanduser(fig_path))
        plt.clf()
    return True


def generate_incidence_plots(site, loss_df, mlp):
    site_df = loss_df[loss_df.Site == site_tokens[site].iloc[0]].reset_index(drop=True)
    mle_df = (
        site_df.sort_values(by=["loss"], ascending=True).head(10).reset_index(drop=True)
    )
    # Eval emulator
    for idx, row in mle_df.iterrows():
        inputs = torch.tensor(
            row.iloc[0:8].values.astype("float"), dtype=torch.float32
        ).reshape(1, 8)
        inputs = normalize(inputs, min_values, max_values, num_features=7)
        outputs = mlp(inputs)
        emulator_outputs_incidence = outputs["Incidence"]
        sim_means = row[row.index.str.contains("mean")][
            dims["Incidence"]["begin_idx"] : dims["Incidence"]["end_idx"]
        ].values
        mle_df_row = pd.DataFrame(
            {
                "Age Bin": senegal_age_bins,
                "Emulator": pd.Series(emulator_outputs_incidence.tolist()[0]),
                "Simulation (Mean)": pd.Series(sim_means),
            }
        )
        ax = sns.lineplot(
            x="Age Bin",
            y="value",
            hue="variable",
            data=pd.melt(mle_df_row, ["Age Bin"]),
            palette=["#009E73", "#0072B2"],
        )
        title = f"{site_names[site]}"
        ax.set(xlabel="Age (years)", ylabel="Annual Clinical Incidence", title=title)
        plt.gca().legend().set_title("")
        fig_path = (
            f"~/EMOD-calibration/emulator/multisite/plots/{site}/incidence_{idx}.png"
        )
        plt.savefig(os.path.expanduser(fig_path))
        plt.clf()
    return True


def generate_parasitemia_plots(site, loss_df, mlp):
    site_df = loss_df[loss_df.Site == site_tokens[site].iloc[0]].reset_index(drop=True)
    mle_df = (
        site_df.sort_values(by=["loss"], ascending=True).head(10).reset_index(drop=True)
    )
    months_for_site = site_metadata[site]["months"]
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
    # Eval emulator
    for idx, row in mle_df.iterrows():
        inputs = torch.tensor(
            row.iloc[0:8].values.astype("float"), dtype=torch.float32
        ).reshape(1, 8)
        inputs = normalize(inputs, min_values, max_values, num_features=7)
        outputs = mlp(inputs)
        emulator_outputs_parasitemia = outputs["Parasitemia_2"][0].detach()
        sim_means = row[row.index.str.contains("mean")][
            dims["Parasitemia_2"]["begin_idx"] : dims["Parasitemia_2"]["end_idx"]
        ].values
        n_row = len(age_labels)
        n_col = len(months_for_site)
        sns.set_style("darkgrid")
        fig, axs = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=(20, 10), constrained_layout=True
        )

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
            sim_data_for_month = sim_means_by_month_dict[month]
            emulator_data_for_month = emulator_means_by_month_dict[month]
            sim_data_by_age_month = np.array_split(
                sim_data_for_month, dims["Parasitemia_2"]["n_age"]
            )
            emulator_data_by_age_month = np.array_split(
                emulator_data_for_month, dims["Parasitemia_2"]["n_age"]
            )
            curr_month = month
            axs[0, i].set_title(calendar.month_name[curr_month])
            for j in range(n_row):
                sns.lineplot(
                    x=x,
                    y=emulator_data_by_age_month[j],
                    label="Emulator",
                    color="#009E73",
                    ax=axs[j, i],
                    legend=False,
                )
                sns.lineplot(
                    x=x,
                    y=sim_data_by_age_month[j],
                    label="Simulation (Mean)",
                    color="#0072B2",
                    ax=axs[j, i],
                    legend=False,
                )
                axs[j, i].set_ylabel(None)
                axs[j, (n_col - 1)].set_ylabel(age_labels[j])
                axs[j, (n_col - 1)].yaxis.set_label_position("right")
                if j != n_row - 1:
                    axs[j, i].set_xticklabels("")
                if i > 0:
                    axs[j, i].set_yticklabels("")
                if i == 0 and j == 0:
                    axs[j, i].legend(loc="upper right")

        fig.suptitle(f"Parasitemia: {site_names[site]}")
        fig_path = (
            f"~/EMOD-calibration/emulator/multisite/plots/{site}/parasitemia_{idx}.png"
        )
        fig.supxlabel("Asexual Parasites per uL")
        fig.supylabel("Fraction of Population")
        plt.setp(axs, ylim=(0, 1))
        plt.savefig(os.path.expanduser(fig_path))
        plt.clf()
    return True


def generate_parasitemia_gametocytemia_plots(site, loss_df, mlp):
    # TODO cleanup
    site_df = loss_df[loss_df.Site == site_tokens[site].iloc[0]].reset_index(drop=True)
    mle_df = (
        site_df.sort_values(by=["loss"], ascending=True).head(10).reset_index(drop=True)
    )
    months_for_site = site_metadata[site]["months"]
    x = ["0", "50", "500", "5000", "50000", "inf"]
    age_labels = [
        "<5 year",
        "5-15 years",
        ">15 years",
    ]
    for idx, row in mle_df.iterrows():
        inputs = torch.tensor(
            row.iloc[0:8].values.astype("float"), dtype=torch.float32
        ).reshape(1, 8)
        inputs = normalize(inputs, min_values, max_values, num_features=7)
        outputs = mlp(inputs)
        emulator_outputs_parasitemia = outputs["Parasitemia_1"][0].detach()
        sim_means = row[row.index.str.contains("mean")][
            dims["Parasitemia_1"]["begin_idx"] : dims["Parasitemia_1"]["end_idx"]
        ].values
        n_row = len(age_labels)
        n_col = len(months_for_site)
        sns.set_style("darkgrid")
        fig, axs = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=(15, 15), constrained_layout=True
        )

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
            sim_data_for_month = sim_means_by_month_dict[month]
            emulator_data_for_month = emulator_means_by_month_dict[month]
            sim_data_by_age_month = np.array_split(
                sim_data_for_month, dims["Parasitemia_1"]["n_age"]
            )
            emulator_data_by_age_month = np.array_split(
                emulator_data_for_month, dims["Parasitemia_1"]["n_age"]
            )
            curr_month = month
            axs[0, i].set_title(calendar.month_name[curr_month])
            for j in range(n_row):
                sns.lineplot(
                    x=x,
                    y=emulator_data_by_age_month[j],
                    label="Emulator",
                    color="#009E73",
                    ax=axs[j, i],
                    legend=False,
                )
                sns.lineplot(
                    x=x,
                    y=sim_data_by_age_month[j],
                    label="Simulation (Mean)",
                    color="#0072B2",
                    ax=axs[j, i],
                    legend=False,
                )
                axs[j, i].set_ylabel(None)
                axs[j, (n_col - 1)].set_ylabel(age_labels[j])
                axs[j, (n_col - 1)].yaxis.set_label_position("right")
                if j != n_row - 1:
                    axs[j, i].set_xticklabels("")
                if i > 0:
                    axs[j, i].set_yticklabels("")
                if i == 0 and j == 0:
                    axs[j, i].legend(loc="upper right")

        fig.suptitle(f"Parasitemia: {site_names[site]}")
        fig_path = (
            f"~/EMOD-calibration/emulator/multisite/plots/{site}/parasitemia_{idx}.png"
        )
        fig.supxlabel("per uL")
        fig.supylabel("Fraction of Population")
        plt.setp(axs, ylim=(0, 1))
        plt.savefig(os.path.expanduser(fig_path))
        plt.clf()

    for idx, row in mle_df.iterrows():
        inputs = torch.tensor(
            row.iloc[0:8].values.astype("float"), dtype=torch.float32
        ).reshape(1, 8)
        inputs = normalize(inputs, min_values, max_values, num_features=7)
        outputs = mlp(inputs)
        emulator_outputs_gametocytemia = outputs["Gametocytemia_1"][0].detach()
        sim_means = row[row.index.str.contains("mean")][
            dims["Gametocytemia_1"]["begin_idx"] : dims["Gametocytemia_1"]["end_idx"]
        ].values
        n_row = len(age_labels)
        n_col = len(months_for_site)
        sns.set_style("darkgrid")
        fig, axs = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=(15, 15), constrained_layout=True
        )

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
            sim_data_for_month = sim_means_by_month_dict[month]
            emulator_data_for_month = emulator_means_by_month_dict[month]
            sim_data_by_age_month = np.array_split(
                sim_data_for_month, dims["Gametocytemia_1"]["n_age"]
            )
            emulator_data_by_age_month = np.array_split(
                emulator_data_for_month, dims["Gametocytemia_1"]["n_age"]
            )
            curr_month = month
            axs[0, i].set_title(calendar.month_name[curr_month])
            for j in range(n_row):
                sns.lineplot(
                    x=x,
                    y=emulator_data_by_age_month[j],
                    label="Emulator",
                    color="#009E73",
                    ax=axs[j, i],
                    legend=False,
                )
                sns.lineplot(
                    x=x,
                    y=sim_data_by_age_month[j],
                    label="Simulation (Mean)",
                    color="#0072B2",
                    ax=axs[j, i],
                    legend=False,
                )
                axs[j, i].set_ylabel(None)
                axs[j, (n_col - 1)].set_ylabel(age_labels[j])
                axs[j, (n_col - 1)].yaxis.set_label_position("right")
                if j != n_row - 1:
                    axs[j, i].set_xticklabels("")
                if i > 0:
                    axs[j, i].set_yticklabels("")
                if i == 0 and j == 0:
                    axs[j, i].legend(loc="upper right")

        fig.suptitle(f"Gametocytemia: {site_names[site]}")
        fig_path = f"~/EMOD-calibration/emulator/multisite/plots/{site}/gametocytemia_{idx}.png"
        # fig.legend(["Emulator", "Simulation (Mean)"], loc="lower left")
        fig.supxlabel("per uL")
        fig.supylabel("Fraction of Population")
        plt.setp(axs, ylim=(0, 1))
        plt.savefig(os.path.expanduser(fig_path))
        plt.clf()
    return True


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
    site_dict = dict(zip(site_tokens.iloc[0].values, site_tokens.columns))
    loss_fn = nn.MSELoss()
    loss_rows = []

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
            outputs = mlp(inputs)
            total_loss = []
            for output_type in dims.keys():
                dim_info = dims[output_type]
                begin_idx = dim_info["begin_idx"]
                end_idx = dim_info["end_idx"]
                head_loss = loss_fn(
                    outputs[output_type],
                    torch.tensor(sim_means[begin_idx:end_idx].values).unsqueeze(0),
                )
                total_loss.append(head_loss)

            row = pd.concat(
                [
                    input_params,
                    pd.Series(
                        {
                            "loss": sum(total_loss).item(),
                        }
                    ),
                    sim_means,
                ]
            )
            loss_rows.append(row)
    loss_df = pd.concat(loss_rows, axis=1).transpose()

    generate_prevalence_plots("namawala_1991", loss_df, mlp)
    for site in ["ndiop_1993", "dielmo_1990"]:
        generate_incidence_plots(site, loss_df, mlp)

    for site in ["matsari_1970", "rafin_marke_1970", "sugungum_1970"]:
        generate_parasitemia_plots(site, loss_df, mlp)

    for site in ["dapelogo_2007", "laye_2007"]:
        generate_parasitemia_gametocytemia_plots(site, loss_df, mlp)
