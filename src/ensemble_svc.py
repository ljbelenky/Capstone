'''This module combines the prediction vector from the pairwise neural networks into a single prediction. A SVC is used because there are only five dimensions, and all dimensions are approximately equally scaled.

The use of a SVC is preferred over taking the argmax of the prediction probabilities because the spread between positive and negative class is not uniform across neural networks.
'''

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib




class art_SVC():
    def __init__(self, df_file, scale):
        df = pd.read_csv(df_file)
        self.scaler_filename = "scaler.save"
        self.scale = scale
        self.X = df[['0','1','2','3','4']]
        targets = df['actuals']
        le = LabelEncoder()
        le.fit(targets)
        self.y = le.transform(targets)
        self.classes = le.classes_


    def split(self):
        self.X_train, X_test, self.y_train, y_test = train_test_split(self.X,self.y)
        self.X_test, self.X_holdout, self.y_test, self.y_holdout = train_test_split(X_test, y_test)
        if self.scale:
            scaler = StandardScaler()
            scaler.fit(self.X_train)
            self.X_train = scaler.transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            self.X_holdout = scaler.transform(self.X_holdout)

            joblib.dump(scaler, self.scaler_filename)

    def predict_new(self,X_as_df):
        self.load()
        scaler = joblib.load(self.scaler_filename)
        X = X_as_df[['0','1','2','3','4']]
        X = scaler.transform(X)
        return self.model.predict(X)


    def save(self,conditions):
        with open('svc_model.pkl', 'wb') as f:
            pickle.dump(conditions['model'], f)

    def load(self):
        with open('svc_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def fit(self, kernel = 'linear', C=1):
        self.model = SVC(kernel=kernel, C=C, class_weight = 'balanced')
        self.model.fit(self.X_train,self.y_train)

    def predict(self,X):
        '''calculate mean vote'''
        return self.model.predict(X)

    def score(self, X, y):
        accuracy = self.model.score(X,y)
        y_true = [self.classes[x] for x in y]
        y_pred = [self.classes[x] for x in self.predict(X)]
        confusion = confusion_matrix(y_true, y_pred, self.classes)
        return accuracy, confusion

def create_svc():
    '''Go through all the steps to create and tune a SVC
    from the master_df.csv results file from predict_pairwise.
    '''
    csv_file = 'art.csv'
    svc = art_SVC(csv_file, scale = True)
    svc.split()
    accuracies = {'linear':[], 'poly':[],'rbf':[],'sigmoid':[]}

    best_accuracy = 0
    for C in np.logspace(0,3,8):
        for k in ['linear', 'poly', 'rbf', 'sigmoid']:
            svc.fit(kernel = k, C=C)
            # svc.predict()
            accuracy, confusion = svc.score(svc.X_test, svc.y_test)
            accuracies[k].append(accuracy)
            print(k, C, 'Accuracy: ',accuracy)
            print('Confusion: \n', confusion)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_conditions = {'model':svc.model,'kernel':k, 'C':C, 'accuracy': accuracy}
    print('Best conditions: ', best_conditions['kernel'],best_conditions['C'],best_conditions['accuracy'])
    return accuracies, best_conditions

if __name__ == '__main__':

    accuracies, best_conditions = create_svc()

    svc.save(best_conditions)
