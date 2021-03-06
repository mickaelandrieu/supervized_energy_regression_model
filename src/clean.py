"""Clean Data : NaN, invalid and outliers."""

import argparse

import pandas as pd

import config
import utils


def clean_data(should_write: bool, drop_score: bool) -> pd.DataFrame:
    """Clean the RAW data provided.

    Args:
        should_write (bool): if True, saves the DataFrame as CSV file.
        drop_score (bool): if True, drops the Energy Star column.

    Returns:
        DataFrame: the cleaned DataFrame
    """
    file_name = "building-energy-benchmarking.csv"
    df2015 = pd.read_csv("{0}2015-{1}".format(config.INPUT, file_name))
    df2016 = pd.read_csv("{0}2016-{1}".format(config.INPUT, file_name))

    df = (
        df2015.pipe(utils.rename_columns)
        .pipe(utils.complete_location_data)
        .pipe(utils.drop_columns)
        .pipe(utils.concat_data, df2016)
        .pipe(utils.keep_non_residential)
        .pipe(utils.clean_variables_names)
        .pipe(utils.remove_duplicates)
        .pipe(utils.remove_null_values)
        .pipe(utils.clean_data)
        .pipe(utils.remove_useless_variables)
    )

    if drop_score:
        df = df.drop(columns="energystar_score", axis="columns")

    if should_write:
        df.set_index("building_id").to_csv("{0}cleaned_data.csv".format(config.INPUT))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean the DataFrame, writing operation is allowed.",
    )

    parser.add_argument(
        "--write",
        type=bool,
        default=True,
        help="Write a CSV file in <INPUT> folder.",
    )

    parser.add_argument(
        "--drop_score",
        type=bool,
        default=False,
        help="Removes Energy Star Score.",
    )

    args = parser.parse_args()
    clean_data(should_write=args.write, drop_score=args.drop_score)
