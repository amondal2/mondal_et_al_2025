import os
import glob


def get_latest_emulator():
    configs_path = os.path.expanduser("~/EMOD-calibration/emulator/multisite/configs/*")
    models_path = os.path.expanduser(
        "~/EMOD-calibration/emulator/multisite/trained_models/*"
    )
    models = glob.glob(models_path)
    configs = glob.glob(configs_path)
    return max(models, key=os.path.getctime), max(configs, key=os.path.getctime)


def get_latest_emulator_ll():
    configs_path = os.path.expanduser("/deprecated/multisite_ll/configs/*")
    models_path = os.path.expanduser("/deprecated/multisite_ll/trained_models/*")
    models = glob.glob(models_path)
    configs = glob.glob(configs_path)
    return max(models, key=os.path.getctime), max(configs, key=os.path.getctime)


def get_latest_emulator_kids():
    configs_path = os.path.expanduser("/deprecated/multisite_kids/configs/*")
    models_path = os.path.expanduser("/deprecated/multisite_kids/trained_models/*")
    models = glob.glob(models_path)
    configs = glob.glob(configs_path)
    return max(models, key=os.path.getctime), max(configs, key=os.path.getctime)


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
