from flask import Flask, render_template, request
import os
import operator
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import random
from sklearn.externals import joblib
import time
import os.path
from ensemble_predictor import Ensemble_Predictor
from jpg_pipeline import jpgPipeline

'''FLASK WEB APP TO PREDICT AND GUESS ART'''

def predict_one_image(filename):
    jpg = jpgPipeline(filename)
    prediction, p_vector  = ep.predict_one(jpg)
    return prediction, p_vector

class svc_predictor():
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

class art_guesser():
    def __init__(self,paintings):
        self.paintings = paintings
        self.app = Flask(__name__,static_url_path='/static')
        self._add_rules()

    def index(self):
        '''Pick a random painting to display'''
        random_painting = paintings.sample(1)
        self.fname = random_painting['files'].iloc[0]
        prediction, p_vector  = predict_one_image(self.fname)
        # if prediction == 'Expressionism':
        #     prediction = svc.predict(p_vector)
        w,h = Image.open(self.fname).size
        scale = 450/h
        h*=scale
        w*=scale
        title = self.fname.split('/')[-1].replace('.jpg','').replace('_',' ')
        artist = self.fname.split('/')[3].replace('_',' ')
        actual = random_painting['actuals'].iloc[0]
        year = self.fname.split('/')[-2]
        self.art = (artist, title, year, actual, prediction, w, h)
        return render_template('index.html', image = self.fname.replace('static/',''), width=w, height=h)

    def result(self):
        artist, title, year, actual, prediction, w, h = self.art
        if prediction == "Abstract": prediction = "Abstract Expressionism"
        if actual == "Abstract": actual = "Abstract Expressionism"
        scale = 300/h
        h *=scale
        w *=scale
        guess = request.args.get('submit', default = '', type = str)
        return render_template('result.html', guess = guess, actual = actual, prediction = prediction, artist = artist, title = title, year = year, width=w, height=h, image=self.fname.replace('static/',''))

    def about(self):
        return render_template('about.html')

    def run(self):
        '''Start Flask app running.  Art Guesser will start when user navigates to index page'''
        self.app.run(host='0.0.0.0',port = 3000, threaded = True,  debug = True)

    def _add_rules(self):
        ''' Add FLASK url rules for various subpages of the website '''
        self.app.add_url_rule('/', 'index', self.index, methods =['GET','POST'])
        self.app.add_url_rule('/result', 'result', self.result, methods =['GET','POST'])
        self.app.add_url_rule('/about', 'about', self.about)

if __name__ == '__main__':
    paintings = pd.read_csv('holdouts.csv')
    ep = Ensemble_Predictor()
    svc = svc_predictor()
    ag = art_guesser(paintings)
    ag.run()
