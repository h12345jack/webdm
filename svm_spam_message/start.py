#!/usr/bin/env python3
#-*- coding: utf-8 -*-


'''
@author: Eadren
@date: 18/12/2017


THIS IS THE MAIN POINT FOR APPLICATION
'''

import sys
import os

import tokenizer, svm_train, svm_predict
import time


training_feature_matrix = None
training_labels = None
test_feature_matrix = None
test_labels = None


if not os.path.exists('data/training_feature_matrix.pkl') \
        or not os.path.exists('data/training_labels.pkl') \
        or not os.path.exists('data/test_feature_matrix.pkl') \
        or not os.path.exists('data/test_labels.pkl'):

    print('Begin building training and test data feature matrix and labels...')
    training_feature_matrix, training_labels, test_feature_matrix, test_labels = tokenizer.transform_feature_space()
    print('Building training and test data feature matrix and labels successfully')
else:
    print('Using cache training data and test data...')
    training_feature_matrix, training_labels, test_feature_matrix, test_labels = tokenizer.get_feature_space()

svm_clf = None

if not os.path.exists('model/svm.pkl'):
    print('Begin svm training...')
    svm_clf = svm_train.train_svm(training_feature_matrix, training_labels) # model save to 'model/svm.pkl' by default
    print('svm training finished successfully')
else:
    svm_clf = svm_train.get_trained_svm()

print('Begin test svm model for test data ... ')
start_time = time.time()
svm_train.test_clf(svm_clf, test_feature_matrix, test_labels) 
end_time = time.time()
print('End testing, the predict time for 100000 data is: %lf' % (end_time - start_time))



print('-------------------------------')


sms = '南洋理财，x万起投，年化收益xx%，保本保息，宝山区牡丹江路xxxx号安信广场x楼xxx~xxx室，联系电话：黄先生xxxxxxxxxxx'

print('Test sms:', sms)
print('Test result:', svm_predict.predict(svm_clf, sms))

