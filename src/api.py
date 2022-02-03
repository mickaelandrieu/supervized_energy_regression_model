"""Basic REST API to expose the predictions of the ML app."""
import joblib
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

import config

app = FastAPI()


class Building(BaseModel):
    """The expected model from REST API request."""

    neighborhood: str
    list_of_all_property_use_types: str
    largest_property_use_type: str
    council_district_code: int
    number_of_floors: int
    energystar_score: float
    distance_to_center: float
    surface_per_building: float
    age: int
    have_parking: bool
    building_primary_type: str


@app.post("/site_energy_use")
def predict(building: Building):
    """Return ML predictions for the Site Energy Use.

    Args:
        building: (Building) the parsed data from user request

    Returns:
        A dictionnary with the predicted target em
        and the related probability
    """
    model = joblib.load("{0}best_model{1}.z".format(config.MODELS, "site_energy_use"))
    # @TODO: build the sample
    sample = {}

    prediction = model.predict([sample])[0]

    return {"prediction": prediction}


@app.post("/emissions")
def predict(building: Building):
    """Return ML predictions for the Building emissions.

    Args:
        building: (Building) the parsed data from user request

    Returns:
        A dictionnary with the predicted target em
        and the related probability
    """
    model = joblib.load("{0}best_model{1}.z".format(config.MODELS, "site_energy_use"))
    # @TODO: build the sample
    sample = {}

    prediction = model.predict([sample])[0]

    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config.DOCKER_HOST,
        port=config.DOCKER_PORT,
        reload=True,
        debug=True,
        workers=3,
    )
