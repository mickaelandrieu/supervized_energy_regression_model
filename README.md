# Predict CO2 emissions and total energy use of Seattle Buildings

## How to get the dataset

You can download the dataset and its documentation on [Kaggle](https://www.kaggle.com/city-of-seattle/sea-building-energy-benchmarking#2015-building-energy-benchmarking.csv)

## Local installation

```bash
python -m venv dev
source dev/Scripts/activate
pip install -r requirements.txt
```

## Docker installation

### Build the image

```bash
docker build --tag app:1.0 .
```

## Train the model

1. Download the RAW data ;
2. Execute `src/clean.py` to create `cleaned_data.csv` ;
3. Execute `src/prepare_features.py` to create `training.pkl` ;
4. Execute `src/create_folds.py` to create `training_folds.pkl` ;
4. Execute `src/tune_hyper_parameters.py` to get optimal parameters ;
5. Execute `src/best.py` to train the model ;

## Evaluate the performance of the models

```bash
python src/report.py --fold=1
```

> fold value is in range [0,4]

## Quality tools

```bash
python -m isort src/
python -m black src/
python -m flake8 src/ --count --statistics
```

## LICENSE

This project is provided under the MIT license.
