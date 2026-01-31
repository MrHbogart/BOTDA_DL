"""
Configuration module for BOTDA Deep Learning project.
Manages paths, constants, and settings for the entire pipeline.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def get_paths(project_name: str = None):
    """
    Get or create directory paths for a specific project.
    
    Args:
        project_name: Name of the project (e.g., 'bgs_rgrs', 'bps_rgrs')
    
    Returns:
        dict: Dictionary containing all path configurations
    """
    # Create directories if they don't exist
    for folder in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, EXPERIMENTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    paths = {
        "data_dir": DATA_DIR,
        "models_dir": MODELS_DIR,
        "results_dir": RESULTS_DIR,
        "logs_dir": LOGS_DIR,
        "experiments_dir": EXPERIMENTS_DIR,
    }

    if project_name:
        project_model_dir = MODELS_DIR / project_name
        project_results_dir = RESULTS_DIR / project_name
        project_logs_dir = LOGS_DIR / project_name
        scalers_dir = project_model_dir / "scalers"

        for folder in [project_model_dir, project_results_dir, project_logs_dir, scalers_dir]:
            folder.mkdir(parents=True, exist_ok=True)

        paths.update({
            "project_name": project_name,
            "model_path": project_model_dir / f"{project_name}_best_model.keras",
            "log_dir": project_logs_dir,
            "results_dir": project_results_dir,
            "scalers_dir": scalers_dir,
        })

    return paths


# BOTDA Data Constants
SHIFT = 10800
FREQUENCY_START_MHZ = 0
FREQUENCY_END_MHZ = 10934 - SHIFT  # Adjusted for shift

# Frequency axis configuration
FREQUENCY_RESOLUTION = 68  # Number of frequency points

# Data types
DATA_TYPES = {
    "BGS": "Brillouin Gain Spectrum",
    "BPS": "Brillouin Phase Spectrum",
}

# Analysis approaches
ANALYSIS_APPROACHES = {
    "RGRS": "Regression (Peak and FWHM estimation)",
    "RGRS_PAPER": "Regression (Paper-based approach)",
    "CLSS": "Classification (Spectral classification)",
}

# Model hyperparameters
MODEL_DEFAULTS = {
    "epochs": 128,
    "batch_size": 2048,
    "test_size": 0.2,
    "random_state": 42,
    "initial_lr": 0.001,
    "early_stopping_patience": 32,
    "reduce_lr_patience": 8,
}

# Synthetic data generation
SYNTHETIC_DATA_DEFAULTS = {
    "n_samples": 300000,
    "noise_std": 0.01,
}

# Visualization settings
VISUALIZATION = {
    "figsize": (15, 5),
    "dpi": 100,
    "style": "whitegrid",
}
