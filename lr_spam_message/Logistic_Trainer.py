# -*- coding: utf-8 -*-
import random
import numpy as np
import json
import time as t
from scipy import sparse, io
from sklearn import metrics
from sklearn.externals import joblib
import sklearn.feature_extraction.text
from sklearn.linear_model import LogisticRegression
from initial_tfidf import create_word_vec
from data_import import load_data
from data_import import read_word_vec

class Logistic_Trainer:

    def __init__(self, training_data, training_label):
        self.training_data = training_data
        self.training_label = training_label
        self.clf = LogisticRegression()
        
    def train(self):
        self.clf.fit(self.training_data, self.training_label)
        joblib.dump(self.clf, 'model/logistic_estimator.pkl')
        training_result = self.clf.predict(self.training_data)
        print (metrics.classification_report(self.training_label, training_result))

if '__main__' == __name__:
    training_data, training_label = create_word_vec()
    t0 = t.time()
    #training_data, training_label = read_word_vec()
    print("initial finished")
    trainer = Logistic_Trainer(training_data, training_label)
    trainer.train()
    print("predicting time: %0.3fs" % (t.time() - t0))
