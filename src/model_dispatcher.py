"""List all regression ML models used for this project."""
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

base = DummyRegressor()
linear = LinearRegression(n_jobs=-1)
poly_linear = Pipeline(
    [
        ("poly", PolynomialFeatures()),
        ("linear", LinearRegression(fit_intercept=False)),
    ]
)
raw_svm = LinearSVR(dual=False, loss="squared_epsilon_insensitive")
decision_tree = DecisionTreeRegressor()
random_forest = RandomForestRegressor(n_jobs=-1)

models = {
    "base": base,
    "linear_reg": linear,
    "poly_linear_reg": poly_linear,
    "raw_svm": raw_svm,
    "decision_tree": decision_tree,
    "random_forest": random_forest,
}
