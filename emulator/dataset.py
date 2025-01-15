"""
Abstraction over emulator input data for use in NN training.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


def get_dataframe(input_file, site=None):
    full_df = pd.read_pickle(input_file)
    site_tokens = pd.read_csv(
        "~/EMOD-calibration/emulator/site_tokens.csv", index_col=0
    )
    if site:
        token = site_tokens[site].iloc[0]
        site_df = full_df.loc[full_df["Site"] == token]
        return site_df
    return full_df


class EmulatorDataset(Dataset):
    def __init__(self, input_file, site=None):
        self.frame = get_dataframe(input_file, site)
        self.n_params = 8
        self.out_dims = 884

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        input_vals = torch.tensor(
            np.float32(row[0 : self.n_params]), dtype=torch.float32
        )
        sim_id = row["sim_id"]
        output_vals = torch.tensor(
            np.float32(row[(self.n_params + 1) :]),
            dtype=torch.float32,
        )
        return input_vals, output_vals, sim_id


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))
