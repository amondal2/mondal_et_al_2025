import calendar
import itertools
import os
from collections import OrderedDict

import pandas as pd
import logging
from scipy.special import gammaln
import numpy as np

from calibration.config import site_metadata

logger = logging.getLogger(__name__)


def aggregate_on_index(df, index, keep=slice(None)):
    """
    Aggregate and re-index data on specified (multi-)index (levels and) intervals
    :param df: a pandas.DataFrame with columns matching the specified (Multi)Index (level) names
    :param index: pandas.(Multi)Index of categorical values or right-bin-edges, e.g. ['early', 'late'] or [5, 15, 100]
    :param keep: optional list of columns to keep, default=all
    :return: pandas.Series or DataFrame of specified channels aggregated and indexed on the specified binning
    """

    if isinstance(index, pd.MultiIndex):
        levels = index.levels
    else:
        levels = [
            index
        ]  # Only one "level" for Index. Put into list for generic pattern as for MultiIndex

    for ix in levels:

        if ix.name not in df.columns:
            raise Exception(
                "Cannot perform aggregation as MultiIndex level (%s) not found in DataFrame:\n%s"
                % (ix.name, df.head())
            )

        # If dtype is object, these are categorical (e.g. season='start_wet', channel='gametocytemia')
        if ix.dtype == "object":
            # TODO: DatetimeIndex, TimedeltaIndex are implemented as int64.  Do we want to catch them separately?
            # Keep values present in reference index-level values; drop any that are not
            df = df[df[ix.name].isin(ix.values)]

        # Otherwise, the index-level values are upper-edges of aggregation bins for the corresponding channel
        elif ix.dtype in ["int64", "float64"]:
            bin_edges = np.concatenate(([-np.inf], ix.values))
            labels = ix.values  # just using upper-edges as in reference
            # TODO: more informative labels? would need to be modified also in reference to maintain useful link...
            # labels = []
            # for low, high in pairwise(bin_edges):
            #     if low == -np.inf:
            #         labels.append("<= {0}".format(high))
            #     elif high == np.inf:
            #         labels.append("> {0}".format(low))
            #     else:
            #         labels.append("{0} - {1}".format(low, high))

            df[ix.name] = pd.cut(df[ix.name], bin_edges, labels=labels)

        else:
            print("")
    # Aggregate on reference MultiIndex, keeping specified channels and dropping missing data
    if keep != slice(None):
        df = df.groupby([ix.name for ix in levels]).sum()[keep].dropna()
    return df


def grouped_df_date(df, pfprdict, index, column_keep, column_del):
    """
    Recut dataframe to recategorize data into desired age and parasitemia bins

    df - Dataframe to be rebinned

    pfprdict - Dictionary mapping postive counts per slide view (http://garkiproject.nd.edu/demographic-parasitological-surveys.html)
                to density of parasites/gametocytes per uL

    index - Multi index into which 'df' is rebinned

    column_keep - Column (e.g. parasitemia) to keep

    column_del - Column (e.g. gametocytemia) to delete
    """
    dftemp = df.copy()
    del dftemp[column_del]

    dftemp["PfPR Bin"] = df[column_keep]
    dftemp = aggregate_on_index(dftemp, index)

    dfGrouped = dftemp.groupby(["Date", "Age Bin", "PfPR Bin"])

    dftemp = dfGrouped[column_keep].count()
    dftemp = dftemp.unstack().fillna(0).stack()
    dftemp = dftemp.rename(column_keep).reset_index()
    dftemp["PfPR Bin"] = [pfprdict[p] for p in dftemp["PfPR Bin"]]

    dftemp = dftemp.set_index(["Date", "Age Bin", "PfPR Bin"])

    return dftemp


def channel_age_json_to_pandas(reference, index_key="Age Bin"):
    """
    A helper function to convert reference data from its form in e.g. site_Dielmo.py:

        reference_data = {
            "Average Population by Age Bin": [ 55, 60, 55, 50, 50, ... ],
            "Age Bin": [ 1, 2, 3, 4, 5, ... ],
            "Annual Clinical Incidence by Age Bin": [ 3.2, 5, 6.1, 4.75, 3.1, ... ]
        }

    To a pd.DataFrame:

             Annual Clinical Incidence by Age Bin  Average Population by Age Bin
    Age Bin
    1                                        3.20                             55
    2                                        5.00                             60
    3                                        6.10                             55
    4                                        4.75                             50
    5                                        3.10                             50

    """
    reference_df = pd.DataFrame(reference)
    reference_df.set_index(index_key, inplace=True)

    logger.debug("\n%s", reference_df)
    return reference_df


def gamma_poisson_likelihood(emulator_df, reference_df):
    LL = (
        gammaln(reference_df.Observations + emulator_df.Observations + 1)
        - gammaln(reference_df.Observations + 1)
        - gammaln(emulator_df.Observations + 1)
    )

    ix = reference_df.Trials > 0
    LL.loc[ix] += (reference_df.loc[ix].Observations + 1) * np.log(
        reference_df.loc[ix].Trials
    )

    ix = emulator_df.Trials > 0
    LL.loc[ix] += (emulator_df.loc[ix].Observations + 1) * np.log(
        emulator_df.loc[ix].Trials
    )

    ix = (reference_df.Trials > 0) & (emulator_df.Trials > 0)
    LL.loc[ix] -= (
        reference_df.loc[ix].Observations + emulator_df.loc[ix].Observations + 1
    ) * np.log(reference_df.loc[ix].Trials + emulator_df.loc[ix].Trials)

    return LL.mean()


def beta_binomial_likelihood(emulator_df, reference_df):
    LL = (
        gammaln(reference_df.Trials + 1)
        + gammaln(emulator_df.Trials + 2)
        - gammaln(reference_df.Trials + emulator_df.Trials + 2)
        + gammaln(reference_df.Observations + emulator_df.Observations + 1)
        + gammaln(
            reference_df.Trials
            - reference_df.Observations
            + emulator_df.Trials
            - emulator_df.Observations
            + 1
        )
        - gammaln(reference_df.Observations + 1)
        - gammaln(reference_df.Trials - reference_df.Observations + 1)
        - gammaln(emulator_df.Observations + 1)
        - gammaln(emulator_df.Trials - emulator_df.Observations + 1)
    )
    return LL.mean()


def dirichlet_multinomial_likelihood(emulator_df, reference_df, n_categories):
    LL = (
        gammaln(reference_df.Observations.values + 1)
        + gammaln(emulator_df.Observations.values)
        - gammaln(
            reference_df.Observations.values
            + emulator_df.Observations.values
            + n_categories
        )
        + gammaln(reference_df.Trials.values + emulator_df.Trials.values + 1)
        - gammaln(emulator_df.Trials.values + 1)
        - gammaln(reference_df.Trials.values + 1)
    )

    return LL.mean() / n_categories


def calculate_trials(row):
    return row["Counts_tot"].values[0]


def calculate_observations(row):
    """
    Get counts of infections per (month,age) grouping
    """
    return row.loc[row.index.get_level_values("PfPR Bin") != 0].sum()["Counts"]


def get_garki_reference(metadata=None):
    reference_csv = os.path.expanduser(
        "~/EMOD-calibration/reference_datasets/Garki_df_dates.csv"
    )
    df = pd.read_csv(reference_csv)
    df = df.loc[df["Village"] == metadata["village"]]

    pfprBinsDensity = metadata["parasitemia_bins"]
    uL_per_field = 0.5 / 200.0  # from Garki PDF - page 111 - 0.5 uL per 200 views
    pfprBins = 1 - np.exp(-np.asarray(pfprBinsDensity) * uL_per_field)
    pfprdict = dict(zip(pfprBins, pfprBinsDensity))

    mask = (df["Date"] > metadata["start_date"]) & (df["Date"] < metadata["end_date"])
    df = df.loc[mask]

    bins = OrderedDict(
        [
            ("Date", list(df["Date"])),
            ("Age Bin", metadata["age_bins"]),
            ("PfPR Bin", pfprBins),
        ]
    )
    bin_tuples = list(itertools.product(*bins.values()))
    index = pd.MultiIndex.from_tuples(bin_tuples, names=bins.keys())

    df = df.rename(columns={"Age": "Age Bin"})

    df2 = grouped_df_date(df, pfprdict, index, "Parasitemia", "Gametocytemia")
    df3 = grouped_df_date(df, pfprdict, index, "Gametocytemia", "Parasitemia")
    dfJoined = df2.join(df3).fillna(0)
    dfJoined = pd.concat([dfJoined["Gametocytemia"], dfJoined["Parasitemia"]])
    dfJoined.name = "Counts"
    dftemp = dfJoined.reset_index()
    dftemp["Channel"] = "Smeared PfPR by Gametocytemia and Age Bin"
    dftemp.loc[len(dftemp) / 2 :, "Channel"] = "Smeared PfPR by Parasitemia and Age Bin"
    dftempsumlist = list(dftemp.groupby(["Channel", "Date", "Age Bin"])["Counts"].sum())
    dftemplist = [[s] * len(dftemp["PfPR Bin"].unique()) for s in dftempsumlist]
    dftemp["Counts_tot"] = [item for sublist in dftemplist for item in sublist]
    dftemp["Counts"] = (
        dftemp.groupby(["Channel", "Date", "Age Bin"])["Counts"]
        .apply(lambda x: x / float(x.sum()))
        .reset_index(drop=True)
    )
    dftemp = dftemp.set_index(["Channel", "Date", "Age Bin", "PfPR Bin"])

    dftemp = dftemp.reset_index()
    dftemp["Date"] = pd.to_datetime(dftemp["Date"]).dt.strftime("%b")
    temppop = dftemp.groupby(["Channel", "Date", "Age Bin", "PfPR Bin"])[
        "Counts_tot"
    ].apply(np.sum)
    dftemp = (
        dftemp.groupby(["Channel", "Date", "Age Bin", "PfPR Bin"])["Counts"]
        .apply(np.mean)
        .reset_index()
    )
    dftemp["Counts_tot"] = list(temppop)
    dftemp["Date"] = dftemp["Date"].apply(lambda x: list(calendar.month_abbr).index(x))
    dftemp = dftemp.sort_values(by=["Channel", "Date"])
    reference = dftemp.set_index(["Channel", "Date", "Age Bin", "PfPR Bin"])
    # Only parasitemia for Garki
    reference = reference.loc[
        reference.index.get_level_values("Channel")
        == "Smeared PfPR by Parasitemia and Age Bin"
    ]
    # Reformatting for likelihood calculations
    likelihood_df = reference.copy(deep=True)
    likelihood_df["Counts"] = likelihood_df["Counts"] * likelihood_df["Counts_tot"]
    trials_obs_df = pd.DataFrame()
    trials_obs_df["Trials"] = likelihood_df.groupby(
        ["Channel", "Date", "Age Bin"]
    ).apply(lambda x: calculate_trials(x))
    trials_obs_df["Observations"] = likelihood_df.groupby(
        ["Channel", "Date", "Age Bin"]
    ).apply(lambda x: calculate_observations(x))

    # trials_obs_df = trials_obs_df.loc[
    #     trials_obs_df.index.get_level_values(level="Age Bin").isin([8, 18, 28, 43, 70])
    # ]
    #
    # reference = reference.loc[
    #     reference.index.get_level_values(level="Age Bin").isin([8, 18, 28, 43, 70])
    # ]

    return {
        "likelihood": trials_obs_df.dropna(),
        "prevalence": reference,
    }


def json_to_pandas(channel_data, bins, channel=None):
    """
    A function to convert nested array channel data from a json file to
    a pandas.Series with the specified MultiIndex binning.
    """

    logger.debug(
        "Converting JSON data from '%s' channel to pandas.Series with %s MultiIndex.",
        channel,
        bins.keys(),
    )
    bin_tuples = list(itertools.product(*bins.values()))
    multi_index = pd.MultiIndex.from_tuples(bin_tuples, names=bins.keys())

    channel_series = pd.Series(
        np.array(channel_data).flatten(), index=multi_index, name=channel
    )
    logger.debug("\n%s", channel_series)
    return channel_series


def season_channel_age_density_json_to_pandas(reference, bins):
    """
    A helper function to convert reference data from its form in e.g. site_Laye.py::

        "Seasons": {
            "start_wet": {
                "PfPR by Parasitemia and Age Bin": [
                    [2, 0, 0, 0, 1, 1], [4, 1, 2, 3, 2, 6], [7, 9, 4, 2, 4, 1]],
                "PfPR by Gametocytemia and Age Bin": [
                    [0, 0, 0, 5, 0, 0], [3, 9, 8, 1, 0, 0], [16, 4, 6, 1, 0, 0]]
            },
            ...
        }

    To a pd.DataFrame with Multi Index::

        Channel                            Season     Age Bin   PfPR Bin      Counts
        PfPR by Gametocytemia and Age Bin  start_wet  5         0             0
                                                                50            0
                                                                500           0
                                                                5000          5
                                                                50000         0
                                                                500000        0
        ...

    """

    season_dict = {}
    for season, season_data in reference.items():
        channel_dict = {}
        for channel, channel_data in season_data.items():
            channel_dict[channel] = json_to_pandas(channel_data, bins)
        season_dict[season] = pd.DataFrame(channel_dict)

    # Concatenate the multi-channel (i.e. parasitemia, gametocytemia) dataframes by season
    df = pd.concat(
        season_dict.values(),
        axis=1,
        keys=season_dict.keys(),
        names=["Season", "Channel"],
    )

    # Stack the hierarchical columns into the MultiIndex
    channel_series = (
        df.stack(["Season", "Channel"])
        .reorder_levels(["Channel", "Season", "Age Bin", "PfPR Bin"])
        .sort_index()
    )

    reference_df = pd.DataFrame(
        channel_series.rename("Counts")
    )  # 1-column DataFrame for standardized combine/compare
    logger.debug("\n%s", reference_df)

    return reference_df


def get_bf_reference(metadata=None):
    reference_bins = OrderedDict(
        [("Age Bin", metadata["age_bins"]), ("PfPR Bin", metadata["parasitemia_bins"])]
    )
    reference_data = season_channel_age_density_json_to_pandas(
        metadata["reference_dict"], reference_bins
    ).reset_index()
    reference_data = reference_data.rename(columns={"Season": "Date"})
    reference_data["Date"] = reference_data["Date"].apply(
        lambda x: list(metadata["seasons_by_month"].keys())[
            list(metadata["seasons_by_month"].values()).index(x)
        ]
    )
    reference_data["Date"] = reference_data["Date"].apply(
        lambda x: list(calendar.month_name).index(x)
    )
    reference_data = reference_data.sort_values(
        ["Channel", "Date", "Age Bin", "PfPR Bin"]
    )
    counts_tot_list = list(
        reference_data.groupby(["Channel", "Date", "Age Bin"])["Counts"].apply(np.sum)
    )
    counts_tot = [[a] * len(metadata["parasitemia_bins"]) for a in counts_tot_list]
    reference_data["Counts_tot"] = [item for sublist in counts_tot for item in sublist]
    reference_data["Counts"] = (
        reference_data.groupby(["Channel", "Date", "Age Bin"])["Counts"]
        .apply(lambda x: x / float(x.sum()))
        .reset_index(drop=True)
    )
    reference = reference_data.set_index(["Channel", "Date", "Age Bin", "PfPR Bin"])

    # Reformatting for likelihood calculations
    likelihood_df = reference.copy(deep=True)
    likelihood_df["Counts"] = likelihood_df["Counts"] * likelihood_df["Counts_tot"]
    trials_obs_df = pd.DataFrame()
    trials_obs_df["Trials"] = likelihood_df.groupby(
        ["Channel", "Date", "Age Bin"]
    ).apply(lambda x: calculate_trials(x))
    trials_obs_df["Observations"] = likelihood_df.groupby(
        ["Channel", "Date", "Age Bin"]
    ).apply(lambda x: calculate_observations(x))

    return {
        "likelihood": trials_obs_df.dropna(),
        "prevalence": reference,
    }


def assemble_site_reference_data():
    """
    Use config to create vector of all reference data.
    """

    # Namawala
    namawala_data = np.array(site_metadata["namawala_1991"]["PfPr"])

    # Ndiop, Dielmo
    ndiop_data = np.array(site_metadata["ndiop_1993"]["Incidence"])
    min_value = np.min(ndiop_data)
    max_value = np.max(ndiop_data)
    ndiop_data_scaled = (ndiop_data - min_value) / (max_value - min_value)

    dielmo_data = np.array(site_metadata["dielmo_1990"]["Incidence"])
    min_value = np.min(dielmo_data)
    max_value = np.max(dielmo_data)
    dielmo_data_scaled = (dielmo_data - min_value) / (max_value - min_value)

    # Dapelogo, Laye
    dapelogo_data = get_bf_reference(metadata=site_metadata["dapelogo_2007"])
    laye_data = get_bf_reference(metadata=site_metadata["laye_2007"])

    # Matsari, Rafin Marke, Sugungum

    matsari_data = get_garki_reference(metadata=site_metadata["matsari_1970"])
    rafin_marke_data = get_garki_reference(metadata=site_metadata["rafin_marke_1970"])
    sugungum_data = get_garki_reference(metadata=site_metadata["sugungum_1970"])

    # Add to dictionary as well for easier plotting
    data_dict_likelihood = {
        "namawala_1991": namawala_data,
        "ndiop_1993": ndiop_data_scaled,
        "dielmo_1990": dielmo_data_scaled,
        "dapelogo_2007": dapelogo_data["likelihood"],
        "laye_2007": laye_data["likelihood"],
        "matsari_1970": matsari_data["likelihood"],
        "rafin_marke_1970": rafin_marke_data["likelihood"],
        "sugungum_1970": sugungum_data["likelihood"],
        "dielmo_1990_unscaled": dielmo_data,
        "ndiop_1993_unscaled": ndiop_data,
    }

    data_dict_prevalence = {
        "namawala_1991": namawala_data,
        "ndiop_1993": ndiop_data_scaled,
        "dielmo_1990": dielmo_data_scaled,
        "dapelogo_2007": dapelogo_data["prevalence"],
        "laye_2007": laye_data["prevalence"],
        "matsari_1970": matsari_data["prevalence"],
        "rafin_marke_1970": rafin_marke_data["prevalence"],
        "sugungum_1970": sugungum_data["prevalence"],
        "dielmo_1990_unscaled": dielmo_data,
        "ndiop_1993_unscaled": ndiop_data,
    }

    return data_dict_likelihood, data_dict_prevalence
