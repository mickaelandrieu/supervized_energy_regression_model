"""Clean Data : NaN, invalid and outliers."""

import argparse

import pandas as pd

import config
import utils


def clean_data(should_write: bool) -> pd.DataFrame:
    """Clean the RAW data provided.

    Args:
        should_write (bool): if True, saves the DataFrame as CSV file.

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
        default=False,
        help="Write a CSV file in <INPUT> folder.",
    )

    args = parser.parse_args()
    clean_data(should_write=args.write)
