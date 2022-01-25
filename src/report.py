"""Report performance of every model in one table."""

import argparse

from time import time

import numpy as np
import pandas as pd

from rich.console import Console
from rich.table import Table
from rich.text import Text
from sklearn import metrics

import config
import model_dispatcher

console = Console()


def run(fold):
    """Run the fold on the list of models.

    Args:
        fold: (int) the selected fold.
    """
    df = pd.read_pickle(config.TRAINING_FOLDS_FILE)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = np.array(df_train.drop(config.TARGET, axis="columns"))
    y_train = np.array(df_train[config.TARGET])

    x_valid = np.array(df_valid.drop(config.TARGET, axis="columns"))
    y_valid = np.array(df_valid[config.TARGET])

    title = Text("Target for this report is the following: ``{0}``".format(config.TARGET), style="bold blue")
    console.print(title)

    report = Table(title="Models performance for fold n°{0}".format(fold))
    report.add_column("Model", style="cyan")
    report.add_column("Time (s)", style="magenta")
    
    for metric in config.METRICS:
        report.add_column(metric, style="spring_green2")

    for name in model_dispatcher.models.keys():
        clf = model_dispatcher.models[name]

        start = time()
        clf.fit(x_train, y_train)
        preds = clf.predict(x_valid)
        
        results = {}
        for metric in config.METRICS:
            metric_func = getattr(metrics, metric)
        
            results[metric] = str(round(metric_func(y_valid, preds), 5))
        elapsed_time = str(round(time() - start, 3))


        report.add_row(name, elapsed_time, *results.values())

    console.print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display a table with all the models on a specific fold."
    )

    parser.add_argument("--fold", type=int, default=0, help="The fold [0, 4].")

    args = parser.parse_args()
    run(fold=args.fold)
