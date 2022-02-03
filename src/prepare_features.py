"""Feature selection and preparation."""
# @TODO: this scripts is built using the insights from EDA

import argparse

import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

import config
import utils


def feat_engineer_data(should_write: bool) -> pd.DataFrame:
    """Feat engineer the cleaned data provided.

    Args:
        should_write (bool): if True, saves the DataFrame as PKL file.

    Returns:
        DataFrame: the cleaned DataFrame
    """
    df = pd.read_csv("{0}cleaned_data.csv".format(config.INPUT))

    df = (
        df.pipe(utils.create_variables)
        .pipe(utils.transform_target, config.TARGET)
        .pipe(utils.create_target_stats, config.TARGET, config.INPUT)
        .pipe(utils.remove_no_business_value_variables)
        .pipe(utils.select_best_features, config.TARGET)
        .pipe(pd.DataFrame.dropna)
        .pipe(utils.encode_categorical)
        .pipe(utils.apply_scaling)
    )

    pd.DataFrame(df.columns).to_csv("{}prepared_features.csv".format(config.DOCS))

    if should_write:
        df.to_pickle("{0}".format(config.TRAINING_FILE))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature engineering on the DataFrame, writing operation is allowed.",
    )

    parser.add_argument(
        "--write",
        type=bool,
        default=True,
        help="Write a CSV file in <INPUT> folder.",
    )

    args = parser.parse_args()
    feat_engineer_data(should_write=args.write)
