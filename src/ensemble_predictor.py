import pickle
import os
from keras.models import load_model
import numpy as np

class Ensemble_Predictor():
    def __init__(self):
        with open('models.pkl','rb') as pkl:
            self.model_dict = pickle.load(pkl)
        root = '../pairwise/'
        for key, value in self.model_dict.items():
            model_path = os.path.join(root,key,value['mfile'])
            value['model'] = load_model(model_path)

    def predict(self,jpgX):
        predictions = {
            'Abstract':[],
            'Cubism':[],
            'Expressionism':[],
            'Pointillism':[]}
        p_vector = []
        for k, value in self.model_dict.items():
            X = jpgX[value['size']]
            result1 = value['model'].predict(X)[0,0]
            result0 = 1-result1
            predictions[value['zero_class']].append(result0)
            predictions[value['one_class']].append(result1)
            p_vector.append(result1)

        combined_predictions = {style:np.mean(values) for style, values in predictions.items()}
        predicted_style = max(combined_predictions, key=lambda key:combined_predictions[key])
        anti_style = min(combined_predictions, key=lambda key:combined_predictions[key])
        return predicted_style, anti_style, predictions, p_vector

    def predict_one(self,jpg):
        prediction, anti_prediction, prob_dictionary, p_vector = self.predict(jpg.X)
        return prediction, p_vector
