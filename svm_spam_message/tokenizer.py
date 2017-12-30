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
from file_helper import (TRAINING_DATA_PATH, TEST_DATA_PATH, TRAINING_FEATURE_MATRIX_PATH, 
TRAINING_LABEL_PATH, TEST_FEATURE_MATRIX_PATH, TEST_LABEL_PATH)


class TfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        def analyzer(doc):
            words = pseg.cut(doc)
            new_doc = ''.join(w.word for w in words if w.flag != 'x')
            #new_doc = ''.join(w.word for w in words if (w.flag != 'x') and (w.flag != 'eng'))
            words = jieba.cut(new_doc)
            return words
        return analyzer

def read_data(path=TRAINING_DATA_PATH):
    with open(path) as f:
        data = f.read().split('\n')[:-1] # remove last line
    
    labels = list(map(lambda x: x[0], data))
    message_data = list(map(lambda x: x.split('\t')[1].strip(), data))
    return message_data, labels

def transform_feature_space():
    training_message_data, training_labels = read_data(TRAINING_DATA_PATH)
    test_message_data, test_labels = read_data(TEST_DATA_PATH)
    
    training_len = len(training_message_data)

    tfidf_vz = TfidfVectorizer(min_df=2, max_df=0.8)
    feature_matrix = tfidf_vz.fit_transform(training_message_data + test_message_data)
    training_feature_matrix = feature_matrix[:training_len]
    test_feature_matrix = feature_matrix[training_len:]

    joblib.dump(tfidf_vz, TFIDF_VZ_PATH)
    joblib.dump(training_feature_matrix, TRAINING_FEATURE_MATRIX_PATH) 
    joblib.dump(training_labels, TRAINING_LABEL_PATH)
    joblib.dump(test_feature_matrix, TEST_FEATURE_MATRIX_PATH) 
    joblib.dump(test_labels, TEST_LABEL_PATH)

    return training_feature_matrix, training_labels, test_feature_matrix, test_labels


def get_feature_space():
    training_feature_matrix = joblib.load(TRAINING_FEATURE_MATRIX_PATH)
    training_labels = joblib.load(TRAINING_LABEL_PATH)
    test_feature_matrix = joblib.load(TEST_FEATURE_MATRIX_PATH)
    test_labels = joblib.load(TEST_LABEL_PATH)

    return training_feature_matrix, training_labels, test_feature_matrix, test_labels

