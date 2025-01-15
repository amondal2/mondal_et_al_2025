import pandas as pd

n_samples = 10000
df_full = pd.read_pickle("scaled_df.pkl")
df_sample = df_full.sample(n_samples)
df_sample.to_pickle("scaled_df_sample.pkl")
