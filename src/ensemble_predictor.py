import numpy as np
from PIL import Image
import os
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import ensemble_svc
import operator
from ensemble_svc import art_SVC
import pickle
from sklearn.externals import joblib

class jpgPipeline():
    def __init__(self,filename,with_target = False, image_mode = 'RGB'):
        self.fname = filename
        self.image_mode =image_mode
        self._load()
        # self._find_target(with_target)


    def _load(self):
        self.image = Image.open(self.fname).convert(mode = self.image_mode)
        box = self.image.getbbox()
        image = self.image.crop(box)
        w = box[2]-box[0]
        h = box[3]-box[1]
        a = min(w,h)*.9
        left = (w-a)/2
        right = (w+a)/2
        top = (h-a)/2
        bottom = (h+a)/2
        cropped_image = image.crop((left,top,right,bottom))
        large_thumbnail=cropped_image.resize((200,200))
        small_thumbnail=cropped_image.resize((150,150))
        large = np.expand_dims(np.asarray(large_thumbnail)/255,axis = 0)
        small = np.expand_dims(np.asarray(small_thumbnail)/255, axis = 0)
        self.X = {'large':large,'small':small}

    def _find_target(self, with_target):
        if with_target:
            self.y = 'target dummy'
        else:
            self.y = None

class Ensemble_Predictor():
    def __init__(self):
        self.model_dict = {
        'Abstract-Cubism':{
            'mfile':'A-C_final_large.mdl',
            'zero_class':'Abstract',
            'one_class':'Cubism',
            'model':None,
            'size':'large'},
        'Abstract-Expressionism':{
            'mfile':'checkpoint.mdl',
            'zero_class':'Abstract',
            'one_class':'Expressionism',
            'model':None,
            'size':'small'},
        'Abstract-Pointillism':{
            'mfile':'checkpoint_large.mdl',
            'zero_class':'Abstract',
            'one_class':'Pointillism',
            'model':None,
            'size':'large'},
        'Cubism-Expressionism':{
            'mfile':'checkpoint_large.mdl',
            'zero_class':'Cubism',
            'one_class':'Expressionism',
            'model':None,
            'size':'large'},
        'Pointillism-Cubism':{
            'mfile':'final_model_large.mdl',
            'zero_class':'Cubism',
            'one_class':'Pointillism',
            'model': None,
            'size':'large'}}

        self._load_models()

    def _load_models(self):
        root = '../pairwise/'
        for key, value in self.model_dict.items():
            model_path = os.path.join(root,key,value['mfile'])
            value['model'] = load_model(model_path)

        with open('svc_model.pkl', 'rb') as f:
            self.svc = pickle.load(f)
        self.scaler = joblib.load("scaler.save")

    def predict(self,jpgX):
        predictions = {
            'Abstract':[],
            'Cubism':[],
            'Expressionism':[],
            'Pointillism':[]}
        p_vector = []
        for k, value in self.model_dict.items():
            X = jpgX[value['size']]  #Allows some models to use 200x200, and others to use 150x150
            result1 = value['model'].predict(X)[0,0]
            result0 = 1-result1
            predictions[value['zero_class']].append(result0)
            predictions[value['one_class']].append(result1)
            p_vector.append(result1)

        combined_predictions = {style:np.mean(values) for style, values in predictions.items()}
        predicted_style = max(combined_predictions, key=lambda key:combined_predictions[key])
        anti_style = min(combined_predictions, key=lambda key:combined_predictions[key])
        if predicted_style == 'Expressionism':
            svc_vector = np.exapnd_dims(p_vector,axis=0)
            scaled_vector = self.scaler.transform(svc_vector)
            prediction = self.svc.predict(scaled_vector)
            print('***********************')
            print(prediction)
            print('***********************')
        return predicted_style, anti_style, predictions, p_vector

    def probabilities(self,predictions):
        result = []
        for k,v in predictions.items():
            result.extend(v)
        return result

    def predict_one(self,jpg):
        # try:
        prediction, anti_prediction, prob_dictionary, p_vector = self.predict(jpg.X)
        return prediction, p_vector
        # except:
        #     print('Unfortunately, this file appears to be corrupted.\nPlease select another')





def predict_on_all_images():

    styles = ['Abstract','Cubism','Pointillism','Expressionism']
    predictions= []
    actuals = []
    files_hist = []
    p_vectors = []
    for style in styles:
        file_list = []
        for root,dirs,files in os.walk('static/images/'+style):
            for file in files:
                file_list.append(os.path.join(root,file))


        for filename in file_list:
            try:
                jpg = jpgPipeline(filename)
                prediction, anti_prediction, prob_dictionary, p_vector = ep.predict(jpg.X)
                print('actual: {}, prediction: {}'.format(style,prediction))
                predictions.append(prediction)
                actuals.append(style)
                files_hist.append(filename)
                p_vectors.append(p_vector)
            except:
                print(filename, ' appears to be corrupted')
    df = pd.DataFrame({'predictions':predictions, 'actuals':actuals,'files':files_hist})
    pvector_df = pd.DataFrame(p_vectors)
    master_df = pd.concat([df,pvector_df], axis = 1)
    master_df.to_csv('master_df.csv')




def predict_one_image(filename):
    ep = Ensemble_Predictor()
    root_dir = '../images/'
    jpg = jpgPipeline(root_dir+filename)
    prediction, p_vector  = ep.predict_one(jpg)
    print(filename, prediction)

if __name__ == '__main__':
    #predict_on_all_images()
    ep = Ensemble_Predictor()
    filename = 'Pointillism/Georges_Seurat/1882-1883/Farm_Women_at_Work.jpg'
    # predict_one_image(filename)
