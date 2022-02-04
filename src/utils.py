"""List of functions used for this project."""

import ast

import haversine as hs
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename variables from the 2015 CSV file.

    Args:
        df (pd.DataFrame): The DataFrame from 2015 CSV file

    Returns:
        pd.DataFrame: A DataFrame
    """
    return df.rename(
        columns={
            "Comment": "Comments",
            "GHGEmissions(MetricTonsCO2e)": "TotalGHGEmissions",
            "GHGEmissionsIntensity(kgCO2e/ft2)": "GHGEmissionsIntensity",
        },
    )


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that doesnt exists in 2016 data.

    Args:
        df (pd.DataFrame): The DataFrame

    Returns:
        pd.DataFrame: A DataFrame
    """
    return df.drop(
        columns=[
            "SPD Beats",
            "Seattle Police Department Micro Community Policing Plan Areas",
            "2010 Census Tracts",
            "OtherFuelUse(kBtu)",
            "City Council Districts",
            "Location",
            "Zip Codes",
        ],
    )


def concat_data(df: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Concat 2015 and 2016 Data.

    Args:
        df (pd.DataFrame): The 2015 DataFrame
        df2 (pd.DataFrame): The 2016 DataFrame

    Returns:
        pd.DataFrame: A DataFrame
    """
    return pd.concat([df, df2], axis="rows")


def complete_location_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Data from Location field in 2015 Data.

    Args:
        df (pd.DataFrame): The DataFrame

    Returns:
        pd.DataFrame: A DataFrame
    """
    return df.apply(extract_data_from_location, axis="columns")


def extract_data_from_location(row: pd.Series) -> pd.Series:
    """Extract from Location variable more information.

    Args:
        row (pd.Series): The DataFrame row

    Returns:
        pd.Series: A DataFrame row
    """
    building = row.copy()
    parsed_location = ast.literal_eval(building.Location)
    building["Latitude"] = parsed_location["latitude"]
    building["Longitude"] = parsed_location["longitude"]
    parsed_human_address = ast.literal_eval(parsed_location["human_address"])
    building["Address"] = parsed_human_address["address"]
    building["City"] = parsed_human_address["city"]
    building["State"] = parsed_human_address["state"]
    building["ZipCode"] = parsed_human_address["zip"]

    return building


def clean_variables_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename and lowercase every variable.

       Also rename some of them, like the targets.

    Args:
        df (pd.DataFrame): The DataFrame

    Returns:
        pd.DataFrame: A DataFrame
    """
    regexp = r"(?<!^)(?=[A-Z])"
    df.columns = df.columns.str.strip().str.replace(regexp, "_", regex=True).str.lower()

    return df.rename(
        columns={
            "o_s_e_building_i_d": "building_id",
            "numberof_buildings": "number_of_buildings",
            "numberof_floors": "number_of_floors",
            "property_g_f_a_total": "property_gfa_total",
            "property_g_f_a_parking": "property_gfa_parking",
            "property_g_f_a_building(s)": "property_gfa_building",
            "largest_property_use_type_g_f_a": "largest_property_use_type_gfa",
            "second_largest_property_use_type_g_f_a": "second_largest_property_use_type_gfa",
            "third_largest_property_use_type_g_f_a": "third_largest_property_use_type_gfa",
            "years_e_n_e_r_g_y_s_t_a_r_certified": "years_energy_star_certified",
            "e_n_e_r_g_y_s_t_a_r_score": "energystar_score",
            "site_e_u_i(k_btu/sf)": "site_eui",
            "site_e_u_i_w_n(k_btu/sf)": "site_euiwn",
            "source_e_u_i(k_btu/sf)": "source_eui",
            "source_e_u_i_w_n(k_btu/sf)": "source_euiwn",
            "site_energy_use(k_btu)": "site_energy_use_target",
            "site_energy_use_w_n(k_btu)": "site_energy_use_wn",
            "steam_use(k_btu)": "steam_use",
            "electricity(k_wh)": "electricity_kwh",
            "electricity(k_btu)": "electricity",
            "natural_gas(therms)": "natural_gas_therms",
            "natural_gas(k_btu)": "natural_gas",
            "total_g_h_g_emissions": "emissions_target",
            "g_h_g_emissions_intensity": "emissions_intensity",
        },
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Data (huge function that needs to be splited)

    Args:
        df (pd.DataFrame): the DataFrame

    Returns:
        pd.DataFrame: the DataFrame
    """
    df["neighborhood"] = df["neighborhood"].str.lower()
    df = df[~df.site_energy_use_target.isna()]
    df = df[~df.emissions_target.isna()]
    df = df[df.compliance_status == "Compliant"]

    # treat latitude and longitude as floats
    df["latitude"] = df["latitude"].astype("float")
    df["longitude"] = df["longitude"].astype("float")

    SEATTLE_COORDS = [47.606, -122.332]
    seattle_coords = tuple(SEATTLE_COORDS)
    df["coords"] = list(zip(df["latitude"], df["longitude"]))

    df["distance_to_center"] = df.coords.apply(
        lambda x: distance_from(x, seattle_coords)
    )

    return df


def keep_non_residential(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the non residential buildings.

    Args:
        df (pd.DataFrame): The DataFrame

    Returns:
        pd.DataFrame: A DataFrame
    """
    return df[
        ~df["BuildingType"].isin(
            ["Multifamily LR (1-4)", "Multifamily MR (5-9)", "Multifamily HR (10+)"]
        )
    ]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """When concatenated, 2015 and 2016 Data duplicates exists.

    For numerical variables we use the mean, else we keep 2016 Data.

    Args:
        df (pd.DataFrame): The DataFrame

    Returns:
        pd.DataFrame: A DataFrame
    """
    MEAN_YEAR = 2015.5
    numerical_variables = list(df.select_dtypes("number"))
    unique_buildings = df[numerical_variables].groupby("building_id").mean()
    unique_buildings["is_agregation"] = unique_buildings["data_year"] == MEAN_YEAR

    deduplicated_buildings = df.sort_values("data_year").drop_duplicates(
        subset=["building_id"], keep="last"
    )
    numerical_variables.remove("building_id")

    deduplicated_buildings = deduplicated_buildings.drop(
        numerical_variables, axis="columns"
    )
    return deduplicated_buildings.merge(unique_buildings, on="building_id", how="left")


def remove_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove variables with no or a very few values.

    Args:
        df (pd.DataFrame): The DataFrame

    Returns:
        pd.DataFrame: A DataFrame
    """
    return df.drop(
        [
            "comments",
            "outlier",
            "years_energy_star_certified",
            "third_largest_property_use_type_gfa",
            "third_largest_property_use_type",
        ],
        axis="columns",
    )


def remove_useless_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Remove variables with no business value for the subject.

    Args:
        df (pd.DataFrame): The DataFrame

    Returns:
        pd.DataFrame: A DataFrame
    """
    return df.drop(
        ["city", "state", "tax_parcel_identification_number"], axis="columns"
    )


# https://github.com/JamesIgoe/GoogleFitAnalysis/blob/master/Analysis.ipynb


def corr_filter(x: pd.DataFrame, bound: float) -> pd.DataFrame:
    """List only variable with correlation higher than the selected bound.

    Args:
        x (pd.DataFrame): the DataFrame
        bound (float): the value of correlation

    Returns:
        pd.DataFrame: A DataFrame
    """
    xCorr = x.corr()
    xFiltered = xCorr[((xCorr >= bound) | (xCorr <= -bound)) & (xCorr != 1.000)]
    return xFiltered


def corr_filter_flattened(x: pd.DataFrame, bound: float) -> pd.DataFrame:
    """Flatten the DataFrame form corrFilter function to remove NaN values.

    Args:
        x (pd.DataFrame): the DataFrame
        bound (float): the bound as previously described

    Returns:
        pd.DataFrame: the DataFrame
    """
    xFiltered = corr_filter(x, bound)
    xFlattened = xFiltered.unstack().sort_values().drop_duplicates()
    return xFlattened


def filter_for_labels(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Get the list of variables that needs to be removed regarding a specific target.

    Args:
        df (pd.DataFrame): the DataFrame of correlations, see corrFilterFlattened()
        label (str): the name of the variable

    Returns:
        pd.DataFrame: the DataFrame
    """
    df = df.sort_index()

    try:
        sideLeft = df[
            label,
        ]
    except:
        sideLeft = pd.DataFrame()

    try:
        sideRight = df[:, label]
    except:
        sideRight = pd.DataFrame()

    if sideLeft.empty and sideRight.empty:
        return pd.DataFrame()
    elif sideLeft.empty:
        concat = (
            sideRight.to_frame(name="correlation")
            .rename_axis("variable")
            .reset_index(level=0)
        )
        return concat
    elif sideRight.empty:
        concat = (
            sideLeft.to_frame(name="correlation")
            .rename_axis("variable")
            .reset_index(level=0)
        )
        return concat
    else:
        concat = pd.concat([sideLeft, sideRight], axis=1)
        concat["correlation"] = concat[0].fillna(0) + concat[1].fillna(0)
        concat.drop(columns=[0, 1], inplace=True)

        return concat.rename_axis("variable").reset_index(level=0)


def fix_multi_colinearity(df: pd.DataFrame, bound: float, target: str) -> pd.DataFrame:
    """Remove every variable with high correlation with the target (overfitting)

    Args:
        df (pd.DataFrame): The DataFrame
        bound (float): The bound
        target (str): The selected target

    Returns:
        pd.DataFrame: A DataFrame
    """
    corr_df = corr_filter_flattened(df, bound)

    variables_to_remove = filter_for_labels(corr_df, target)["variable"].tolist()

    return df.drop(variables_to_remove, axis="columns")


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Transform every categorical variable to numerical

    Args:
        df (pd.DataFrame): the DataFrame

    Returns:
        pd.DataFrame: the DataFrame
    """

    cols = df.select_dtypes(include=["object"]).columns.tolist()
    df[cols] = df[cols].apply(LabelEncoder().fit_transform)

    return df


def remove_no_business_value_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Theses variables don't have relationship with the target

    Args:
        df (pd.DataFrame): the DataFrame

    Returns:
        pd.DataFrame: the DataFrame
    """

    return df.drop(
        [
            "building_id",
            "property_name",
            "default_data",
            "compliance_status",
            "site_eui",
            "site_euiwn",
            "source_euiwn",
            "source_eui",
            "emissions_intensity",
            "steam_use",
            "natural_gas",
            "natural_gas_therms",
            "second_largest_property_use_type_gfa",
            "second_largest_property_use_type",
            "building_type",
            "primary_property_type",
            "property_gfa_parking",
            "coords",
            "latitude",
            "longitude",
            "address",
            "data_year",
            "is_agregation",
            "zip_code",
            "year_built",
        ],
        axis="columns",
    )


def apply_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Apply scaling to every columns

    Args:
        df (pd.DataFrame): the DataFrame

    Returns:
        pd.DataFrame: the DataFrame
    """
    return pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)


def distance_from(coords1: tuple, coords2: tuple) -> float:
    """Calculate the distance from coordinates

    Args:
        coords1 (tuple): a tuple (lat, long)
        coords2 (tuple): a tuple (lat, long)

    Returns:
        float: the distance
    """
    return round(hs.haversine(coords1, coords2), 2)


def transform_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Create a new column with a np.log of the selected variable

    Args:
        df (pd.DataFrame): A DataFrame
        target (str): the selected variable

    Returns:
        pd.DataFrame: A DataFrame with a new variable
    """
    df[target] = df[df[target] > 0][target]
    df[target] = np.log(df[target])

    return df


def create_variables(df: pd.DataFrame) -> pd.DataFrame:
    df["number_of_floors"] = df["number_of_floors"] + 1
    df["surface_per_floor"] = df["property_gfa_building"] / df["number_of_floors"]
    df["surface_per_building"] = df["property_gfa_building"] / df["number_of_buildings"]
    df["age"] = 2022 - df["year_built"]
    df["have_parking"] = df["property_gfa_parking"] > 0
    df["building_primary_type"] = (
        df["building_type"] + " " + df["primary_property_type"]
    )

    return df


def select_best_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df[
        [
            "surface_per_building",
            "building_primary_type",
            "target_mean",
            "have_parking",
            "council_district_code",
            "distance_to_center",
            "age",
            "surface_per_floor",
            "energystar_score",
            target,
        ]
    ]


def create_target_stats(df: pd.DataFrame, target: str, path: str) -> pd.DataFrame:
    """Generate targets statistics per Building Primary Type

    Args:
        df (pd.DataFrame): A DataFrame

    Returns:
        pd.DataFrame: A DataFrame
    """
    aggs = {}
    aggs[target] = ["max", "min", "mean", "count", "std"]
    stats = df[[target, "building_primary_type"]]
    stats = stats.groupby("building_primary_type").agg(aggs)
    stats = stats.fillna(0).reset_index()

    stats.T.reset_index(drop=True).T
    stats.columns = [
        "building_primary_type",
        "max",
        "min",
        "target_mean",
        "count",
        "target_std",
    ]

    stats.to_csv("{0}target_stats.csv".format(path))

    return df.merge(
        stats[["building_primary_type", "target_mean"]], on="building_primary_type"
    )
