"""Extracts the documentation from JSON metadata."""
import json

import pandas as pd

import config


def write_xls_docs():
    """Write Variables documentation in a human readable way."""
    variables = []
    for year in (2015, 2016):
        file_path = "{0}{1}-metadata.json".format(config.DOCS, year)
        with open(file_path) as metadata:
            json_content = json.load(metadata)
            columns = json_content.get("columns")

            for column in columns:
                variables.append(
                    {
                        "name": column.get("name"),
                        "type": column.get("dataTypeName"),
                        "description": column.get("description"),
                        "year": year,
                    },
                )

    df = pd.DataFrame(variables)

    df.to_excel("{0}variables.xlsx".format(config.DOCS))


if __name__ == "__main__":
    write_xls_docs()
