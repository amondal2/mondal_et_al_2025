"""
Generate dataframe of sites and associated EIR and interventions.
"""

import pandas as pd
import os

if __name__ == "__main__":
    eir_filepath = os.path.expanduser(
        "~/EMOD-calibration/simulation_inputs/monthly_eirs/eir_by_site.csv"
    )
    eir_df = pd.read_csv(eir_filepath)
    eir_df = eir_df.T.iloc[:, :-1]
    eir_df = eir_df.drop("month")
    eir_df.columns = [f"eir_{i + 1}" for i in eir_df.columns]
    eir_df = eir_df.reset_index(names="Site")

    # Add intervention boolean
    sim_coord_filepath = os.path.expanduser(
        "~/EMOD-calibration/simulation_inputs/simulation_coordinator.csv"
    )
    coord_df = pd.read_csv(sim_coord_filepath)
    coord_df = coord_df[["site", "CM_filepath"]]
    coord_df["CM_filepath"] = coord_df["CM_filepath"].notnull().astype("float")
    coord_df = coord_df.rename(
        columns={"site": "Site", "CM_filepath": "has_interventions"}
    )
    full_df = eir_df.merge(coord_df, on="Site")
    full_df.to_csv(
        os.path.expanduser(
            "~/EMOD-calibration/simulations/site_eirs_and_interventions.csv"
        )
    )
