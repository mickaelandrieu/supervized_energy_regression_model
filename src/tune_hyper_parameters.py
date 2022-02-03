"""Improve and configure the selected model using GridSearchCV."""

import argparse

import numpy as np
import pandas as pd

from rich.console import Console
from rich.table import Table
from sklearn import metrics, model_selection

import config
import model_dispatcher

console = Console()

random_forest = model_dispatcher.models["random_forest"]

param_grid = {
    "n_estimators": [120, 300, 500, 800, 1200],
    "criterion": ["squared_error", "absolute_error"],
    "max_depth": [5, 8, 15, 25, 30, None],
    "min_samples_split": [2, 5, 10, 15, 100],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", "log2", None],
}


def run():
    """Run the folds on the selected model.

    Args:
        model: (str)the selected model to be trained.
    """
    grid = model_selection.RandomizedSearchCV(
        estimator=random_forest,
        param_distributions=param_grid,
        scoring="r2",
        n_iter=100,
        n_jobs=4,
        cv=5,
        verbose=5,
    )

    df = pd.read_pickle(config.TRAINING_FOLDS_FILE)

    x_train = np.array(df.drop([config.TARGET, "kfold"], axis="columns"))
    y_train = np.array(df[config.TARGET])

    grid.fit(x_train, y_train)
    best_score = grid.best_score_

    report = Table(
        title="Best parameters for {0} ({1})".format("Random Forest", best_score)
    )
    report.add_column("Parameter", style="cyan")
    report.add_column("Best value", style="magenta")

    best_params = grid.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        report.add_row(param_name, str(best_params[param_name]))

    console.print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Output the Best parameters for the Random Forest.",
    )

    run()
