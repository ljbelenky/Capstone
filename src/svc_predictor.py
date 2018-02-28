import pickle
from sklearn.externals import joblib
import numpy as np

class SVC_Predictor():
    def __init__(self):
        with open('svc_model.pkl', 'rb') as f:
            self.svc = pickle.load(f)
        self.scaler = joblib.load("scaler.save")

    def predict(self,p_vector):
        p_vector = np.expand_dims(p_vector, axis =0)
        scaled_vector = self.scaler.transform(p_vector)
        prediction = self.svc.predict(scaled_vector)
        predictions = ['Abstract','Cubism','Expressionism','Pointillism']
        return predictions[prediction]
