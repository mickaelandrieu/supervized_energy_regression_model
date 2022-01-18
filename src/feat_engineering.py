"""Feature selection and preparation."""
# @TODO: this scripts is built using the insights from EDA

import argparse

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import config
import utils


def feat_engineer_data(should_write: bool) -> pd.DataFrame:
    """Feat engineer the cleaned data provided.

    Args:
        should_write (bool): if True, saves the DataFrame as PKL file.

    Returns:
        DataFrame: the cleaned DataFrame
    """
    df = pd.read_csv('{0}cleaned_data.csv'.format(config.INPUT))
    
    df_columns = df.columns

    # STUPID 1: Drop all NaN values
    df = df.dropna()
    
    # STUPID 2 : Transform all categorical using Label Encoding
    cols = df.select_dtypes(include=['object']).columns.tolist()
    df[cols] = df[cols].apply(LabelEncoder().fit_transform)
    
    # STUPID 3 : Standard scale everything
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df_columns)

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
        default=False,
        help="Write a CSV file in <INPUT> folder.",
    )

    args = parser.parse_args()
    feat_engineer_data(should_write=args.write)


