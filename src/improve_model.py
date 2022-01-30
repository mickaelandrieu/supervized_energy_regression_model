"""Improve and configure the selected model using GridSearchCV."""

import argparse

from os import path

import joblib
import numpy as np
import pandas as pd

from rich.console import Console
from sklearn import metrics

import config
import model_dispatcher

console = Console()


def run(model):
    """Run the fold on the selected model.

    Args:
        model: (str)the selected model to be trained.
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select a model and output the best parameters.",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="See src/model_dispatcher file.",
    )

    args = parser.parse_args()
    run(fold=args.fold, model=args.model)
