#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''
@author: Eadren
@date: 18/12/2017

INPUT: feature sparse matrix, labels
OUTPUT: training svm model
'''
import os
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from file_helper import SVM_MODEL_PATH

def train_svm(feature_matrix, labels, save_model_path=SVM_MODEL_PATH):

    svm_clf = svm.LinearSVC()
    svm_clf.fit(feature_matrix, labels)
    joblib.dump(svm_clf, save_model_path)

    return svm_clf

def get_trained_svm(model_path=SVM_MODEL_PATH):
    return joblib.load(model_path)


def test_clf(clf, test_data, test_labels):
    predict_result = clf.predict(test_data)
    print("Classification report for classifier %s:\n%s\n" % (
        clf, metrics.classification_report(test_labels, predict_result, digits=4)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predict_result))
    print(precision_recall_fscore_support(test_labels, predict_result))

