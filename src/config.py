"""Stores the project configuration."""

TARGET = "site_energy_use_target"
BOUND = 0.7
TRAINING_FILE = "./input/training.pkl"
TRAINING_FOLDS_FILE = "./input/training_folds.pkl"

INPUT = "./input/"
MODELS = "./models/"
DOCS = "./docs/"

DOCKER_PORT = 8501
DOCKER_HOST = "0.0.0.0"

METRICS = ['max_error', 'mean_squared_error', 'r2_score']
