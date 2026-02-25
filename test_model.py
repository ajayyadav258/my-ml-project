import joblib
import numpy as np
from sklearn.datasets import load_iris

def test_model():
    model = joblib.load("model.pkl")
    iris = load_iris()
    new_data = iris.data[0].reshape(1, -1)
    prediction = model.predict(new_data)
    print("Predicted class:", prediction)