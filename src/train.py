"""Train the data againsts multiple models."""

import argparse

from os import path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rich.console import Console
from rich.table import Table
from sklearn import metrics

import config
import model_dispatcher

console = Console()


def run(model):
    """Run the selected model.

    Args:
        model: (str)the selected model to be trained.
    """
    df = pd.read_pickle(config.TRAINING_FOLDS_FILE)
    columns = df.columns

    for fold in range(5):
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        x_train = np.array(df_train.drop([config.TARGET, "kfold"], axis="columns"))
        y_train = np.array(df_train[config.TARGET])

        x_valid = np.array(df_valid.drop([config.TARGET, "kfold"], axis="columns"))
        y_valid = np.array(df_valid[config.TARGET])

        clf = model_dispatcher.models[model]

        clf.fit(x_train, y_train)

        preds = clf.predict(x_valid)
        importances = clf.feature_importances_
        idxs = np.argsort(importances)

        report = Table(title="Feature importances for {0}".format(model))
        report.add_column("Feature", style="cyan")
        report.add_column("Importance", style="magenta")

        for i in idxs[::-1]:
            report.add_row(columns[i], str(importances[i]))

        console.print(report)

        for metric in config.METRICS:
            metric_func = getattr(metrics, metric)

            results = metric_func(y_valid, preds)
            console.log("Fold={0}, {1}={2}".format(fold, metric, results))

        filepath = path.join("{0}{1}_{2}.z".format(config.MODELS, model, fold))
        joblib.dump(clf, filepath, compress=("zlib", 7))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Data using a model.",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="See src/model_dispatcher file.",
    )

    args = parser.parse_args()
    run(model=args.model)
