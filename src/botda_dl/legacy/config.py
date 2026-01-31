import os

SHIFT = 10800
FREQUENCY_START_MHZ = 0
FREQUENCY_END_MHZ = 10934 - SHIFT

def get_paths():
    pwd = os.getcwd()

    for folders in ["data", "models", "results", "logs"]:
        dir = os.path.join(pwd, folders)
        os.makedirs(dir, exist_ok=True)

    DATA_DIR = os.path.join(pwd, "data")
    MODELS_DIR = os.path.join(pwd, "models")
    SCALERS_DIR= os.path.join(MODELS_DIR, "scalers")
    RESULTS_DIR = os.path.join(pwd, "results")
    LOGS_DIR = os.path.join(pwd, "logs")

    PATHS_DICT = {}
    for data in ['bgs', 'bps']:
        for approach in ['rgrs', 'rgrs_paper']:

            project = f"{data}_{approach}"
            if data == 'bgs':
                data_path = os.path.join(DATA_DIR, "BGS.txt")
            elif data == 'bps':
                data_path = os.path.join(DATA_DIR, "BPS.txt")


            model_path =  MODELS_DIR+f'/{project}_best_model.keras'
            log_dir = LOGS_DIR+f'/{project}_logs'
            results_dir = RESULTS_DIR+f'/{project}_results'
            scalers_dir = SCALERS_DIR+f'/{project}_scalers'

            for dir in [log_dir, results_dir, scalers_dir]:
                os.makedirs(dir, exist_ok=True)

            PATHS_DICT[project] = {
                "data_path": data_path,
                "model_path": model_path,
                "log_dir": log_dir,
                "results_dir": results_dir,
                "scalers_dir": scalers_dir,
            }
    return PATHS_DICT

