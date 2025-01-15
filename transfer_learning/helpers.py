import os
import glob


def get_latest_emulator(exclude_site=None):
    configs_path = os.path.expanduser(
        f"~/EMOD-calibration/transfer_learning/configs/exclude_{exclude_site}/*"
    )
    models_path = os.path.expanduser(
        f"~/EMOD-calibration/transfer_learning/trained_models/exclude_{exclude_site}/*"
    )
    models = glob.glob(models_path)
    configs = glob.glob(configs_path)
    return max(models, key=os.path.getctime, default=""), max(
        configs, key=os.path.getctime, default=""
    )


site_names = {
    "ndiop_1993": "Ndiop",
    "dapelogo_2007": "Dapelogo",
    "dielmo_1990": "Dielmo",
    "matsari_1970": "Matsari",
    "laye_2007": "Laye",
    "rafin_marke_1970": "Rafin Marke",
    "namawala_1991": "Namawala",
    "sugungum_1970": "Sugungum",
}
