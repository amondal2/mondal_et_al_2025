import pandas as pd
import os
import numpy as np
import seaborn as sns
from emulator.helpers import site_names
import matplotlib.pyplot as plt

from simulations import params

plt.style.use("ggplot")
plt.rcParams["pdf.fonttype"] = 42

if __name__ == "__main__":
    sim_df = pd.read_csv(
        os.path.expanduser(
            "~/EMOD-calibration/simulations/download/simulation_runtimes.csv"
        )
    )
    emulator_df = pd.read_csv(
        os.path.expanduser(
            "~/EMOD-calibration/emulator/multisite/emulator_runtimes.csv"
        )
    )
    # Create a 3x3 grid of subplots (8 for sites, 1 empty)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    for i, site in enumerate(params.sites):
        sim_df_site = sim_df[sim_df.Site == site]
        sim_df_site.runtime = np.log(sim_df_site.runtime)
        train_params = [
            "Falciparum_MSP_Variants",
            "Falciparum_Nonspecific_Types",
            "MSP1_Merozoite_Kill_Fraction",
            "Max_Individual_Infections",
            "Falciparum_PfEMP1_Variants",
            "Antigen_Switch_Rate",
            "Nonspecific_Antigenicity_Factor",
        ]
        param_dfs = [y for x, y in sim_df_site.groupby(train_params, as_index=False)]
        sim_means_rows = []
        for param_df in param_dfs:
            row = pd.Series({"runtime": np.mean(param_df["runtime"]), "site": site})
            sim_means_rows.append(row)

        sim_df_site_means = pd.concat(sim_means_rows, axis=1).transpose()
        emulator_df_site = emulator_df[emulator_df.site == site]
        emulator_df_site.runtime = np.log(emulator_df_site.runtime)

        # Plot histograms for both dataframes
        sns.histplot(
            data=sim_df_site_means,
            x="runtime",
            label="Simulation",
            ax=axes[i],
            kde=True,
            color="#CC79A7",
            alpha=0.5,
            stat="probability",
            binwidth=0.5,
            lw=2,
        )
        sns.histplot(
            data=emulator_df_site,
            x="runtime",
            label="Emulator",
            ax=axes[i],
            kde=True,
            color="#009E73",
            alpha=0.5,
            stat="probability",
            binwidth=0.10,
            lw=2,
        )

        # Customize the subplot
        title = f"{site_names[site]}"
        axes[i].grid(True)
        axes[i].margins(x=0, y=0)
        axes[i].set_title(title, fontweight="bold")
        axes[i].set_xlabel("Runtime (s)")
        axes[i].set_ylabel("Proportion of simulations")
        # axes[i].get_legend().remove()
        labels = [item.get_text() for item in axes[i].get_xticklabels()]
        labels = ["4e-5", "6e-3", "1.0", "2e4", "3e6"]
        plt.setp(
            axes[i],
            ylim=(0, 1.0),
            xlim=(
                np.min(emulator_df_site.runtime) - 1.5,
                np.max(np.log(sim_df.runtime)),
            ),
        )
        axes[i].set_xticklabels(labels)
    plt.ticklabel_format(axis="x", style="sci")

    # Remove the last (empty) subplot
    fig.delaxes(axes[-1])

    # Adjust the layout and add a main title
    plt.tight_layout()
    fig.suptitle("Runtime Distributions Comparison Across Sites", fontsize=16, y=1.02)

    fig_path = f"~/EMOD-calibration/emulator/multisite/emulator_runtimes.pdf"
    plt.savefig(os.path.expanduser(fig_path))
    plt.clf()
