import itertools

import pandas as pd
import torch
import os
import json

from calibration.config import site_metadata
from calibration.helpers import assemble_site_reference_data
from emulator.helpers import get_latest_emulator
from emulator.multisite.mlp import MLP
from torcheval.metrics.functional import r2_score


def format_emulator_data_garki(emulator_outputs=None, months=None, dims=None):
    return generate_emulator_df(
        dims=dims,
        output_key="Parasitemia_2",
        channel_name="Smeared PfPR by Parasitemia and Age Bin",
        emulator_outputs=emulator_outputs,
        months=months,
    )


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
    calib_methods = {
        "sgd": "sgd_joint_loss_min.csv",
        "nn": "nearest_neighbors_joint_loss_min.csv",
        "ll": "likelihood_joint_max.csv",
    }
    calib_keys = {"sgd": "loss", "nn": "joint_loss", "ll": "joint_likelihood"}
    calib_fns = {
        "sgd": True,
        "nn": True,
        "ll": False,
    }

    _, data_dict = assemble_site_reference_data()

    device = torch.device("cpu")
    model_path, config_path = get_latest_emulator()
    with open(os.path.expanduser(config_path)) as fp:
        config = json.load(fp)

    site_tokens = pd.read_csv(
        "~/EMOD-calibration/emulator/site_tokens.csv", index_col=0
    )
    site_dict = dict(zip(site_tokens.iloc[0].values, site_tokens.columns))

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
    for method in calib_methods.keys():
        calib_path = os.path.expanduser(
            f"~/EMOD-calibration/calibration/data/{calib_methods[method]}"
        )
        calib_df = pd.read_csv(calib_path)
        calib_df_row = calib_df.sort_values(
            by=[calib_keys[method]], ascending=calib_fns[method]
        ).head(1)
        input_params = [
            "Falciparum_MSP_Variants",
            "Falciparum_Nonspecific_Types",
            "MSP1_Merozoite_Kill_Fraction",
            "Max_Individual_Infections",
            "Falciparum_PfEMP1_Variants",
            "Antigen_Switch_Rate",
            "Nonspecific_Antigenicity_Factor",
        ]
        input_row = calib_df_row[input_params]
        params = torch.tensor(input_row.values, dtype=torch.float, requires_grad=True)
        all_ref = []
        all_outputs = []
        for token, site in site_dict.items():
            if site == "namawala_1991":
                ref = data_dict[site]
                inputs = torch.cat((params.squeeze(), torch.tensor([token]))).unsqueeze(
                    0
                )
                inputs = normalize(inputs, min_values, max_values, num_features=7)
                outputs = mlp(inputs)
                site_outputs = outputs["PfPr"]
                all_ref.append(torch.tensor(ref).unsqueeze(0))
                all_outputs.append(torch.tensor(site_outputs))
            elif site in ["ndiop_1993", "dielmo_1990"]:
                ref = data_dict[site]
                inputs = torch.cat((params.squeeze(), torch.tensor([token]))).unsqueeze(
                    0
                )
                inputs = normalize(inputs, min_values, max_values, num_features=7)
                outputs = mlp(inputs)
                site_outputs = outputs["Incidence"]
                all_ref.append(torch.tensor(ref).unsqueeze(0))
                all_outputs.append(torch.tensor(site_outputs))
            elif site in ["dapelogo_2007", "laye_2007"]:
                ref = data_dict[site]
                inputs = torch.cat((params.squeeze(), torch.tensor([token]))).unsqueeze(
                    0
                )
                inputs = normalize(inputs, min_values, max_values, num_features=7)
                outputs = mlp(inputs)
                emulator_df = format_emulator_data_burkinafaso(
                    outputs, months=site_metadata[site]["months"], dims=dims
                ).sort_values(by=["Age Bin", "Date"])
                reference_df = ref.sort_values(by=["Age Bin", "Date"]).sort_values(
                    by=["Age Bin", "Date"]
                )
                emulator_values = torch.tensor(emulator_df["Counts"].values).unsqueeze(
                    0
                )
                reference_values = torch.tensor(
                    reference_df["Counts"].values, dtype=torch.float32
                ).unsqueeze(0)
                all_outputs.append(emulator_values)
                all_ref.append(reference_values)
            else:
                ref = data_dict[site]
                inputs = torch.cat((params.squeeze(), torch.tensor([token]))).unsqueeze(
                    0
                )
                inputs = normalize(inputs, min_values, max_values, num_features=7)
                outputs = mlp(inputs)
                emulator_df = format_emulator_data_garki(
                    outputs, months=site_metadata[site]["months"], dims=dims
                ).sort_values(by=["Age Bin", "Date"])
                reference_df = ref.sort_values(by=["Age Bin", "Date"])
                emulator_values = torch.tensor(emulator_df["Counts"].values).unsqueeze(
                    0
                )
                reference_values = torch.tensor(
                    reference_df["Counts"].values, dtype=torch.float32
                ).unsqueeze(0)
                all_outputs.append(emulator_values)
                all_ref.append(reference_values)
        all_ref = torch.cat(all_ref, dim=1)
        all_out = torch.cat(all_outputs, dim=1)
        print(f"METHOD >>>> {method}")
        print(f"R2: {r2_score(all_ref.squeeze(), all_out.squeeze())}")
