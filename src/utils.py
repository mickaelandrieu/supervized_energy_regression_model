"""List of functions used for this project."""

import ast

import pandas as pd


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
