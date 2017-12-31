#!/usr/bin/env python3
#-*- coding: utf-8 -*-


'''
@author: Eadren
@date: 18/12/2017

This a tokenizer for SMS implemented by jieba

DATA INPUT: [label] message
DATA OUTPUT: sklearn spare matrix; column is tokens and row is messages
'''

from collections import Counter
import sys 

import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from file_helper import  *


class TfidfVectorizer(TfidfVectorizer):
    """
    tranform each raw message to tf-idf vector

    INPUT: message list
    OUTPUT: tf-idf feature matrix
    """
    def build_analyzer(self):
        def analyzer(doc):
            words = pseg.cut(doc)
            new_doc = ''.join(w.word for w in words if w.flag != 'x')
            #new_doc = ''.join(w.word for w in words if (w.flag != 'x') and (w.flag != 'eng'))
            words = jieba.cut(new_doc)
            return words
        return analyzer

def read_data(path=TRAINING_DATA_PATH):
    """
    read data from given path and  split labels and messages.
   
    INPUT: data path
    OUTPUT: raw message list and corresponding labels
    """
    with open(path) as f:
        data = f.read().split('\n')[:-1] # remove last line
    
    labels = list(map(lambda x: x[0], data))
    message_data = list(map(lambda x: x.split('\t')[1].strip(), data))
    return message_data, labels

def fit_transform_training_feature_space():
    """
    transform all training data tf-idf feature matrix

    INPUT: None
    OUTPUT: training feature matrix, training labels
    """
    training_message_data, training_labels = read_data(TRAINING_DATA_PATH)
    
    tfidf_vz = TfidfVectorizer(min_df=2, max_df=0.8)
    training_feature_matrix = tfidf_vz.fit_transform(training_message_data)

    joblib.dump(tfidf_vz, TFIDF_VZ_PATH)
    joblib.dump(training_feature_matrix, TRAINING_FEATURE_MATRIX_PATH) 
    joblib.dump(training_labels, TRAINING_LABEL_PATH)

    return training_feature_matrix, training_labels


def transform_test_feature_space():
    test_message_data, test_labels = read_data(TEST_DATA_PATH)
    
    tfidf_vz = joblib.load(TFIDF_VZ_PATH)
    test_feature_matrix = tfidf_vz.transform(test_message_data)

    joblib.dump(test_feature_matrix, TEST_FEATURE_MATRIX_PATH) 
    joblib.dump(test_labels, TEST_LABEL_PATH)

    return test_feature_matrix, test_labels



def get_feature_space():
    """
    load feature space from cached/ever transformed data
    

    INPUT: None
    OUTPUT: training feature matrix, training labels
    """
    training_feature_matrix = joblib.load(TRAINING_FEATURE_MATRIX_PATH)
    training_labels = joblib.load(TRAINING_LABEL_PATH)

    return training_feature_matrix, training_labels

