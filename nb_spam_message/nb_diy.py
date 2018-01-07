# coding=utf8
import os
import re
import time

from math import log
from collections import defaultdict
from collections import Counter

import numpy as np

TRIAN_DATA = './data/r8-train-stemmed.txt'
TEST_DATA = './data/r8-test-stemmed.txt'
# TEST_DATA = './data/test.txt'
UNKOWN = '#UNKOWN#'
CATEGORY_UNKOWN = '#CATEGORY_UNKOWN#'

class NBmodel:

    def __init__(self, alpha=1):
        self.alpha = alpha

        self.prob_c = {}
        self.prob_x_c = {}
        self.categorys = []

        self.word_dict = defaultdict(int)

    def encode(self, sentence):
        '''
        将单词进行编码，降低内存消耗
        '''
        w_list = []
        for w in sentence:
            if w not in self.word_dict:
                self.word_dict[w] = len(self.word_dict)
            w_list.append(self.word_dict[w])
        return w_list    

    def decode(self, sentence):
        '''
        将单词进行解码，得到w_list
        '''
        w_list = []
        for w in sentence:
            if w not in self.word_dict:
                w_list.append(UNKOWN)
            else:
                w_list.append(self.word_dict[w])
        return w_list


    def fit(self, X_train, y):
        '''
        X_train为空格隔开的词
        得到类别对应的先验和似然
        '''

        category_dict = defaultdict(int)
        # category_dict[CATEGORY_UNKOWN] = 0
        train_list = []

        for w_list in X_train:
            # 如果是字符串，变成word list
            if isinstance(w_list, str):
                w_list = w_list.split()

            train_list.append(self.encode(w_list))

        for tag in y:
            category_dict[tag] +=1

        # 可能的类别
        self.categorys = category_dict.keys()

        tags_sum = sum(category_dict[tag] for tag in category_dict)

        # probability p(c)
        # prob_c = {tag: category_dict[tag]/tags_sum for tag in category_dict}
        # laplacian correction smooth
        print(len(category_dict), 'category')  # '\frac{D_c + 1}{ D + N}'
        self.prob_c = {tag: (category_dict[tag] + self.alpha) / (tags_sum + self.alpha * len(category_dict))\
                     for tag in category_dict}

        # probablity p(x|c)
        # transform train_list 2 {tag: bag of words}
        category_dict2 = defaultdict(list)
        for tag, content in zip(y, train_list):
            category_dict2[tag].extend(content)

        category_dict2 = {tag: Counter(
            category_dict2[tag]) for tag in category_dict2}
        category_sum = {tag: len(category_dict2[tag]) for tag in category_dict2}
        # prob_x_c = {tag: {w: category_dict2[tag][w]/category_sum[tag] for w in category_dict2[tag] } \
        #                   for tag in category_dict2}
        self.prob_x_c = {tag: {w: (category_dict2[tag][w] + self.alpha) / (category_sum[tag] +  self.alpha * len(category_dict2[tag]))
                          for w in category_dict2[tag]} for tag in category_dict2}
        
        for tag in category_dict2:
            self.prob_x_c[tag][UNKOWN] = 1 / \
                (category_sum[tag] + len(category_dict2[tag]))

    def discriminate(self, w_list, category):
        '''use log instead of product'''
        f = lambda x: self.prob_x_c[category][x] if x in self.prob_x_c[category] else self.prob_x_c[category][UNKOWN]
        v = [log(f(w)) for w in w_list]
        return log(self.prob_c[category]) + sum(v)


    def predict(self, y_test):
        if isinstance(y_test, str):
            y_test = y_test.split()
        w_list = self.decode(y_test)
        v = {c: self.discriminate(w_list, c) for c in self.categorys}
        label = max(v, key=lambda x: v[x])

        return label

def sentence2gram(sentence, ngram_range = (1, 1), analyzer = 'char'):
    '''
    将句子变为ngram的bag of words
    '''
    assert isinstance(sentence, str)

    w_list = sentence.split()
    if analyzer == 'char':
        w_list = [w for w in sentence]

    sentence_bags = []
    for ngram in ngram_range:
        for i in range(0, len(w_list)-ngram):
            sentence_bags.append(''.join([w for w in w_list[i: i+ngram]]))
    return sentence_bags

def test1():
    '''
    =================================================================

    [数据相关](http://ana.cachopo.org/datasets-for-single-label-text-categorization)
    测试的数据集：Reuters-21578 R8， 5485 docs
    =================================================================

    | kNN (k = 10) | 0.8524 |
    | Naive Bayes |  0.9607 |   
    | Centroid (Normalized Sum) | 0.9356 |
    | SVM (Linear Kernel) | 0.9698 |
    
    我的结果:
    total:  2189 right:  2099 accuracy: 0.9588853357697579
    =================================================================

    '''
    nb_model = NBmodel(alpha=1.0)
    X_train = []
    y = []
    with open(TRIAN_DATA) as f:
        for line in f.readlines():
            line = line.strip()
            l_list = re.split(r"[\s\t]", line)
            tag = l_list[0]
            content = l_list[1:-1]
            X_train.append(content)
            y.append(tag)

    nb_model.fit(X_train, y)
    test_list = []
    right = 0
    with open(TEST_DATA) as f:
        for line in f.readlines():
            line = line.strip()
            l_list = re.split(r"[\s\t]", line)
            tag = l_list[0]
            content = l_list[1:-1]
            test_list.append((tag, content))

    for example in test_list:
        right_label = example[0]
        content = example[1]
        my_answer = nb_model.predict(content)
        print(right_label, my_answer)
        if right_label == my_answer:
            right += 1

    print("total: ", len(test_list), "right: ",
          right, 'accuracy:', right / len(test_list))


def test2():
    '''
    =================================================================
    使用diy，不做tf-idf
    total:  99999 right:  99335 accuracy: 0.9933599335993359
                 precision    recall  f1-score   support

              0    0.99656   0.99605   0.99631     89917
              1    0.96495   0.96935   0.96714     10082

    avg / total    0.99337   0.99336   0.99337     99999

    [[89562   355]
     [  309  9773]]
    =================================================================
    非diy，调用sklearn库
    参数：
        cls__alpha: 0.6
        tfidf__norm: 'l2'
        tfidf__sublinear_tf: True
        tfidf__use_idf: True
        vect__max_df: 0.5
        vect__ngram_range: (1, 2)

    total:  99999 right:  99661 accuracy: 0.9966199662
    f1:score 0.983237452886
    total: 
             precision    recall  f1-score   support

          0    0.99812   0.99812   0.99812     89917
          1    0.98324   0.98324   0.98324     10082

    avg / total    0.99662   0.99662   0.99662     99999

    [[89748   169]
     [  169  9913]]
    =================================================================

    '''
    from sklearn.metrics import classification_report,confusion_matrix
    from model import load_data
    CUR_DIR = os.path.abspath(os.path.dirname(__file__))

    TRAIN_DATA = os.path.join(CUR_DIR, './data/training_data.txt')
    TEST_DATA = os.path.join(CUR_DIR, './data/test_data.txt')
    train_data = load_data(TRAIN_DATA)
    test_data = load_data(TEST_DATA)

    X_train = [sentence2gram(i[0]) for i in train_data]
    y_train = [i[1] for i in train_data]

    nb_model = NBmodel(alpha=0.9)
    nb_model.fit(X_train, y_train)
    
    right = 0
    X_test = [sentence2gram(i[0]) for i in test_data]
    y_test = [i[1] for i in test_data]
    y_pred = []
    for test in test_data:
        content = sentence2gram(test[0])
        label = nb_model.predict(content)
        y_pred.append(label)
        if label == test[1]:
            right += 1

    print("total: ", len(test_data), "right: ",
          right, 'accuracy:', right / len(test_data))

    print(classification_report(y_test, y_pred, digits=5))
    print(confusion_matrix(y_test, y_pred))

def main():
    test1()

if __name__ == '__main__':
    main()
