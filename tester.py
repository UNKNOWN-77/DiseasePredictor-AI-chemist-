from joblib import load
import pandas as pd

data = pd.read_csv('training_data.csv')

X = data.iloc[:, 2:].values

pipeline = load("disease_predictor.joblib")

answers = pipeline.predict(X)

