#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''
@author: Eadren
@date: 19/12/2017


INPUT: sms 
OUTPUT: IS SPAM MESSAGE
'''

import json
import jieba
import os
from sklearn.externals import joblib
from file_helper import TFIDF_VZ_PATH

def predict(clf, sms):
    if not os.path.exists(TFIDF_VZ_PATH):
        print('NOT TRAINED, CANNOT PREDICT')
        exit(1)

    tfidf_vz = joblib.load(TFIDF_VZ_PATH)
    tfidf_vector = tfidf_vz.transform([sms])
    predict_result = clf.predict(tfidf_vector)
    print('Predict Finished, result is: ', predict_result[0])
    return predict_result[0] == '1'
