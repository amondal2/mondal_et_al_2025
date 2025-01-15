"""
This script preprocesses input data to prepare it for use in Pytorch.
"""

import pandas as pd
from itertools import chain
from sklearn import preprocessing
import json
import numpy as np


def filter_keys(keys):
    filtered_keys = []
    for key in keys:
        if "(0," in str(key) or "(1," in str(key):
            continue
        else:
            filtered_keys.append(key)
    return filtered_keys


# read in sample dataset
input_df = pd.read_pickle("~/EMOD-calibration/simulations/emulator_input_data_full.pkl")

# Tokenize Sites
sites = input_df["Site"].unique()
le = preprocessing.LabelEncoder()
targets = le.fit_transform(sites)
replace_vals = dict(zip(sites, targets))
input_df["Site"].replace(replace_vals, inplace=True)
pd.DataFrame([replace_vals]).to_csv("site_tokens.csv")

# get number of sites and training parameters to set up dimension of input tensor
num_sites = len(sites)
train_params = [
    "Falciparum_MSP_Variants",
    "Falciparum_Nonspecific_Types",
    "MSP1_Merozoite_Kill_Fraction",
    "Max_Individual_Infections",
    "Falciparum_PfEMP1_Variants",
    "Antigen_Switch_Rate",
    "Nonspecific_Antigenicity_Factor",
]
static_params = ["Site", "sim_id"]

# Scale parameters and save min/max values
param_mins = input_df[train_params].apply(lambda x: x.min(), axis=0)
param_max = input_df[train_params].apply(lambda x: x.max(), axis=0)
param_mins.to_csv("~/EMOD-calibration/simulations/download/param_mins.csv")
param_max.to_csv("~/EMOD-calibration/simulations/download/param_max.csv")

input_df_scaled = input_df[train_params]

input_df_scaled["Site"] = input_df["Site"]
input_df_scaled["sim_id"] = input_df["sim_id"]

# Pull one row to get output sizes
# Keep track of dimensions for plotting later
output_dimensions = {}

row = input_df.head(1)

num_pfpr = len(row["PfPr"].iloc[0])
output_dimensions["PfPr"] = {
    "begin_idx": 0,
    "end_idx": num_pfpr,
    "n_density": num_pfpr,
    "n_age": 1,
    "n_month": 1,
}
begin_idx = num_pfpr
num_incidence = len(row["Incidence"].iloc[0])
output_dimensions["Incidence"] = {
    "begin_idx": begin_idx,
    "end_idx": begin_idx + num_incidence,
    "n_density": num_incidence,
    "n_age": 1,
    "n_month": 1,
}

keys = list(row["Gametocytemia_1"].iloc[0].keys())
n_density = len(row["Gametocytemia_1"].iloc[0][keys[0]])
begin_idx = num_incidence + num_pfpr
num_gametocytemia_1 = n_density * len(keys)
end_idx = begin_idx + num_gametocytemia_1
prefix = "Gametocytemia_1"
n_month = 12
n_age = int(len(keys) / 12)
labels = [f"{prefix}_{i + 1}_{j}" for i in range(n_month) for j in range(n_age)]

curr_idx = begin_idx
for i, l in enumerate(labels):
    output_dimensions[l] = {
        "begin_idx": curr_idx,
        "end_idx": curr_idx + n_density,
        "n_density": n_density,
        "n_age": n_age,
        "n_month": n_month,
    }
    curr_idx = curr_idx + n_density

keys = list(row["Parasitemia_1"].iloc[0].keys())
n_density = len(row["Parasitemia_1"].iloc[0][keys[0]])
begin_idx = end_idx
num_parasitemia_1 = n_density * len(keys)
end_idx = begin_idx + num_parasitemia_1
prefix = "Parasitemia_1"
n_month = 12
n_age = int(len(keys) / 12)
labels = [f"{prefix}_{i + 1}_{j}" for i in range(n_month) for j in range(n_age)]

curr_idx = begin_idx
for _, l in enumerate(labels):
    output_dimensions[l] = {
        "begin_idx": curr_idx,
        "end_idx": curr_idx + n_density,
        "n_density": n_density,
        "n_age": n_age,
        "n_month": n_month,
    }
    curr_idx = curr_idx + n_density

keys = list(row["Parasitemia_2"].iloc[0].keys())
# keys = filter_keys(keys)
keys_par2 = keys
n_density = len(row["Parasitemia_2"].iloc[0][keys[0]])
begin_idx = end_idx
num_parasitemia_2 = n_density * len(keys)
end_idx = begin_idx + num_parasitemia_2
prefix = "Parasitemia_2"
n_month = 12
n_age = int(len(keys) / 12)
labels = [f"{prefix}_{i + 1}_{j}" for i in range(n_month) for j in range(n_age)]

curr_idx = begin_idx
for _, l in enumerate(labels):
    output_dimensions[l] = {
        "begin_idx": curr_idx,
        "end_idx": curr_idx + n_density,
        "n_density": n_density,
        "n_age": n_age,
        "n_month": n_month,
    }
    curr_idx = curr_idx + n_density

with open("output_dimensions.json", "w") as fp:
    json.dump(output_dimensions, fp)

input_df["PfPr_flat"] = input_df["PfPr"]
input_df["Incidence_flat"] = input_df["Incidence"]
input_df["Gametocytemia_1_flat"] = input_df["Gametocytemia_1"].apply(
    lambda z: list(chain.from_iterable(z.values()))
)
input_df["Parasitemia_1_flat"] = input_df["Parasitemia_1"].apply(
    lambda z: list(chain.from_iterable(z.values()))
)


def filter_dict(d):
    return {k: d[k] for k in keys_par2 if k in d}


input_df["Parasitemia_2_flat"] = input_df["Parasitemia_2"].apply(
    lambda z: list(chain.from_iterable(z.values()))
)

# Concat all data
input_df["all_data_flat"] = (
    input_df["PfPr_flat"]
    + +input_df["Incidence_flat"]
    + input_df["Gametocytemia_1_flat"]
    + input_df["Parasitemia_1_flat"]
    + input_df["Parasitemia_2_flat"]
)


def scale_list(prop, x):
    scaled = [prop * i for i in x]
    return scaled


incidence_max = []
incidence_min = []


def scale_incidence(x):
    x = np.array(x)
    x = np.nan_to_num(x)
    min_val = np.min(x)
    max_val = np.max(x)
    scaled = (x - min_val) / (max_val - min_val)
    incidence_min.append(min_val)
    incidence_max.append(max_val)
    return scaled.tolist()


input_df["Incidence_flat_scaled"] = input_df["Incidence_flat"].apply(
    lambda x: scale_incidence(x)
)

incidence_scale_df = pd.DataFrame(
    {
        "sim_id": input_df["sim_id"].tolist(),
        "min_val": incidence_min,
        "max_val": incidence_max,
    }
)
incidence_scale_df.to_csv(
    "~/EMOD-calibration/simulations/download/incidence_scaling.csv"
)

input_df["Gametocytemia_1_flat_scaled"] = input_df["Gametocytemia_1_flat"].apply(
    lambda x: [i for i in x]
)
input_df["Parasitemia_1_flat_scaled"] = input_df["Parasitemia_1_flat"].apply(
    lambda x: [i for i in x]
)
input_df["Parasitemia_2_flat_scaled"] = input_df["Parasitemia_2_flat"].apply(
    lambda x: [i for i in x]
)

input_df["PfPr_flat_scaled"] = input_df["PfPr_flat"].apply(lambda x: [i for i in x])

input_df["all_data_flat_scaled"] = (
    input_df["PfPr_flat_scaled"]
    + input_df["Incidence_flat_scaled"]
    + input_df["Gametocytemia_1_flat_scaled"]
    + input_df["Parasitemia_1_flat_scaled"]
    + input_df["Parasitemia_2_flat_scaled"]
)

full_df = input_df_scaled.merge(
    input_df[["sim_id", "all_data_flat_scaled"]], on="sim_id"
)
full_df = full_df[train_params + static_params].join(
    full_df["all_data_flat_scaled"].apply(pd.Series)
)
full_df = full_df.fillna(0)
full_df.to_pickle("~/EMOD-calibration/emulator/scaled_df.pkl")
