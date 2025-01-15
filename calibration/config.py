"""
Config for calibration.
"""
import os
import numpy as np

base_dir = os.path.expanduser("~/EMOD-calibration/reference_datasets")
senegal_age_bins = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    85,
]

burkinafaso_age_bins = [5, 15, 80]
namawala_age_bins = [1, 2, 4, 5, 10, 15, 20, 30, 40, 50]
garki_age_bins = [1, 4, 8, 18, 28, 43, 70]

site_metadata = {
    "matsari_1970": {
        "parasitemia_bins": [
            0.0,
            16.0,
            70.0,
            409.0,
            np.inf,
        ],  # (, 0] (0, 16] ... (409, inf]
        "age_bins": garki_age_bins,
        "village": "Matsari",
        "end_date": "1971-12-31",
        "start_date": "1970-11-01",
        "filename": "Garki_df_dates.csv",
        "village_label": "Matsari",
        "months": [11, 1, 4, 5, 6, 8],
        "emulator_outcomes": ["Parasitemia_2"],
    },
    "rafin_marke_1970": {
        "emulator_outcomes": ["Parasitemia_2"],
        "filename": "Garki_df_dates.csv",
        "village_label": "Rafin Marke",
        "village": "Rafin Marke",
        "months": [12, 3, 4, 5, 7, 10],
        "age_bins": garki_age_bins,
        "parasitemia_bins": [
            0.0,
            16.0,
            70.0,
            409.0,
            np.inf,
        ],
        "end_date": "1971-12-31",
        "start_date": "1970-11-01",
    },
    "sugungum_1970": {
        "parasitemia_bins": [
            0.0,
            16.0,
            70.0,
            409.0,
            np.inf,
        ],
        "age_bins": garki_age_bins,
        "village": "Sugungum",
        "end_date": "1971-12-31",
        "start_date": "1970-11-01",
        "months": [11, 1, 4, 6, 8],
        "emulator_outcomes": ["Parasitemia_2"],
        "filename": "Garki_df_dates.csv",
    },
    "namawala_1991": {
        "emulator_outcomes": ["PfPr"],
        "village_label": "Namawala",
        "PfPr": [0.85, 0.9, 0.88, 0.85, 0.82, 0.75, 0.65, 0.45, 0.42, 0.4],
        "age_bins": namawala_age_bins,
        "population_by_age_bin": [150, 626, 1252, 626, 2142, 1074, 1074, 605, 605, 489],
    },
    "ndiop_1993": {
        "emulator_outcomes": ["Incidence"],
        "age_bins": senegal_age_bins,
        "site_type": "Senegal",
        "Incidence": [
            1.9,
            2.2,
            2.6,
            2.8,
            2.9,
            3.0,
            2.8,
            2.7,
            2.6,
            2.6,
            2.5,
            2.2,
            2.1,
            1.8,
            1.5,
            1.0,
            0.8,
            0.6,
            0.5,
            0.4,
            0.3,
            0.4,
        ],
        "population_by_age_bin": [
            31,
            34,
            31,
            28,
            28,
            21,
            21,
            21,
            21,
            21,
            15,
            15,
            15,
            15,
            15,
            62,
            42,
            42,
            84,
            39,
            39,
            50,
        ],
    },
    "dielmo_1990": {
        "emulator_outcomes": ["Incidence"],
        "age_bins": senegal_age_bins,
        "Incidence": [
            3.2,
            5,
            6.1,
            4.75,
            3.1,
            2.75,
            2.7,
            1.9,
            0.12,
            0.8,
            0.5,
            0.25,
            0.1,
            0.2,
            0.4,
            0.3,
            0.2,
            0.2,
            0.2,
            0.15,
            0.15,
            0.15,
        ],
        "population_by_age_bin": [
            55,
            60,
            55,
            50,
            50,
            38,
            38,
            38,
            38,
            38,
            26,
            26,
            26,
            26,
            26,
            110,
            75,
            75,
            150,
            70,
            70,
            90,
        ],
    },
    "dapelogo_2007": {
        "parasitemia_bins": [
            0,
            50,
            500,
            5000,
            50000,
            500000,
        ],
        "age_bins": burkinafaso_age_bins,
        "seasons_by_month": {  # Collection dates from raw data in Ouedraogo et al. JID 2015
            "July": "start_wet",  # 29 June - 30 July '07 => [180 - 211]
            "September": "peak_wet",  # 3 Sept - 9 Oct '07 => [246 - 282]
            "January": "end_wet",  # (a.k.a. DRY) 10 Jan - 2 Feb '08 => [10 - 33]
        },
        "village": "Dapelogo",
        "seasons": ["end_wet", "start_wet", "peak_wet"],
        "months": [1, 7, 9],
        "reference_dict": {
            # Digitized by J.Gerardin from data in:
            #   - A.L.Ouedraogo et al. JID 2015
            # for J.Gerardin et al. Malaria Journal 2015, 14:231
            # N.B. the values represent counts of individual observations
            "end_wet": {
                "Smeared PfPR by Parasitemia and Age Bin": [
                    [1, 0, 0, 2, 2, 3],
                    [2, 1, 0, 2, 0, 4],
                    [9, 5, 4, 4, 2, 3],
                ],
                "Smeared PfPR by Gametocytemia and Age Bin": [
                    [0, 1, 4, 2, 2, 0],
                    [0, 1, 4, 6, 0, 0],
                    [12, 8, 3, 4, 0, 0],
                ],
            },
            "start_wet": {
                "Smeared PfPR by Parasitemia and Age Bin": [
                    [1, 2, 0, 1, 3, 1],
                    [2, 5, 2, 3, 1, 1],
                    [6, 8, 4, 4, 0, 2],
                ],
                "Smeared PfPR by Gametocytemia and Age Bin": [
                    [1, 3, 2, 2, 0, 0],
                    [0, 8, 0, 3, 0, 0],
                    [11, 10, 2, 2, 0, 0],
                ],
            },
            "peak_wet": {
                "Smeared PfPR by Parasitemia and Age Bin": [
                    [1, 1, 0, 4, 3, 1],
                    [4, 1, 2, 4, 2, 1],
                    [6, 9, 6, 2, 2, 0],
                ],
                "Smeared PfPR by Gametocytemia and Age Bin": [
                    [2, 3, 2, 2, 1, 0],
                    [2, 5, 4, 2, 1, 0],
                    [14, 7, 4, 0, 0, 0],
                ],
            },
        },
        "emulator_outcomes": ["Parasitemia_1", "Gametocytemia_1"],
    },
    "laye_2007": {
        "parasitemia_bins": [
            0,
            50,
            500,
            5000,
            50000,
            500000,
        ],  # (, 0] (0, 50] ... (50000, ]
        "age_bins": burkinafaso_age_bins,  # (, 5] (5, 15] (15, ]
        "months": [1, 7, 9],
        "seasons_by_month": {  # Collection dates from raw data in Ouedraogo et al. JID 2015
            "July": "start_wet",  # 29 June - 30 July '07 => [180 - 211]
            "September": "peak_wet",  # 3 Sept - 9 Oct '07 => [246 - 282]
            "January": "end_wet",  # (a.k.a. DRY) 10 Jan - 2 Feb '08 => [10 - 33]
        },
        "village": "Laye",
        "seasons": ["end_wet", "start_wet", "peak_wet"],
        "reference_dict": {
            # Digitized by J.Gerardin from data in:
            #   - A.L.Ouedraogo et al. JID 2015
            # for J.Gerardin et al. Malaria Journal 2015, 14:231
            # N.B. the values represent counts of individual observations
            "end_wet": {
                "Smeared PfPR by Parasitemia and Age Bin": [
                    [2, 0, 0, 0, 1, 1],
                    [4, 1, 2, 3, 2, 6],
                    [7, 9, 4, 2, 4, 1],
                ],
                "Smeared PfPR by Gametocytemia and Age Bin": [
                    [0, 0, 0, 5, 0, 0],
                    [3, 9, 8, 1, 0, 0],
                    [16, 4, 6, 1, 0, 0],
                ],
            },
            "start_wet": {
                "Smeared PfPR by Parasitemia and Age Bin": [
                    [0, 1, 0, 1, 1, 0],
                    [13, 1, 0, 3, 0, 1],
                    [9, 12, 3, 0, 1, 0],
                ],
                "Smeared PfPR by Gametocytemia and Age Bin": [
                    [1, 0, 1, 1, 0, 0],
                    [2, 4, 8, 4, 1, 0],
                    [7, 10, 5, 3, 0, 0],
                ],
            },
            "peak_wet": {
                "Smeared PfPR by Parasitemia and Age Bin": [
                    [1, 0, 0, 0, 1, 0],
                    [8, 1, 1, 6, 3, 1],
                    [10, 11, 4, 2, 0, 0],
                ],
                "Smeared PfPR by Gametocytemia and Age Bin": [
                    [1, 0, 0, 1, 0, 0],
                    [7, 9, 3, 1, 0, 0],
                    [14, 10, 3, 0, 0, 0],
                ],
            },
        },
        "emulator_outcomes": ["Parasitemia_1", "Gametocytemia_1"],
    },
}
