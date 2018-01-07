#coding=utf8
import os

import time
import json

from pprint import pprint

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB


CUR_DIR = os.path.abspath(os.path.dirname(__file__))

TRAIN_DATA = os.path.join(CUR_DIR, './data/training_data.txt')
TEST_DATA = os.path.join(CUR_DIR, './data/test_data.txt')
TRAIN_DATA_CUT = os.path.join(CUR_DIR, './data/training_data_wc.txt')

TF_IDF_MODEL = os.path.join(CUR_DIR, './model/tf_idf.pkl')
NB_MODEL = os.path.join(CUR_DIR, './model/nb_model.pkl')


def load_data(data_path):
    '''
    数据读取
    '''
    datas = []
    with open(data_path, encoding='utf8') as f:
        l_num = 0
        for line in f.readlines():
            l_num +=1
            line = line.strip()
            label_and_data = line.split('\t')
            if len(label_and_data) == 2:
                label, data = label_and_data
                datas.append((data, int(label)))
            else:
                print(l_num)
    return datas



def train_test(train_data):
    '''
    tf_idf的特征, naive bayes，
    第一次测试，无调参，最简单的模型
    '''  
    data = [i[0] for i in train_data]
    label = [i[1] for i in train_data]
    pipline = Pipeline([
        ('vect', CountVectorizer(min_df=1, ngram_range=(1,1), decode_error = 'ignore', analyzer='char')),
        ('tfidf', TfidfTransformer())
    ])

    text_fea = pipline.fit_transform(data)
    print(text_fea.shape)
    clf = MultinomialNB()
    
    kfolder = KFold(n_splits=5, random_state=42)
    label = np.array(label)
    for train_index, test_index in kfolder.split(text_fea, label):
        clone_clf = clone(clf)

        x_train_folders = text_fea[train_index]
        y_train_folders = label[train_index]
        
        x_test_folders = text_fea[test_index]
        y_test_folders = label[test_index]

        clone_clf.fit(x_train_folders, y_train_folders)
        y_pred = clone_clf.predict(x_test_folders)

        print("f1:score", f1_score(y_test_folders, y_pred, average='binary'))
        print(classification_report(y_test_folders, y_pred, digits=5))

def alchemist_train(train_data, test_data):
    '''
    炼金与调参
    vect的参数:
    ngram_range: unigrams or bigrams ,trigram is bad
    tfidf的参数:
    use_idf: 是否使用idf
    norm: 范式
    sublinear_tf: 是否使用1 + log(tf)
    cls的参数:
    alpha：(Laplace/Lidstone) smoothing parameter 拉普拉斯平滑参数的alpha
    '''
    data = [i[0] for i in train_data]
    label = [i[1] for i in train_data]
    pipeline = Pipeline([
        ('vect', CountVectorizer(analyzer='char', min_df=1)),
        ('tfidf', TfidfTransformer()),
        ('cls', MultinomialNB())
    ])
    nb_param_grid = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams ,3gram is bad
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1','l2'),
        'tfidf__sublinear_tf': (True, False),
        'cls__alpha': [i/10 for i in range(2, 11, 1)]
    }
   
    grid_search = GridSearchCV(pipeline, nb_param_grid, cv=2, scoring="f1", n_jobs=4, verbose=1)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(nb_param_grid)
    t0 = time.time()
    grid_search.fit(data, label)
    print("done in %0.3fs" % (time.time() - t0))
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(nb_param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    cv_results = grid_search.cv_results_
    grid_search_file = open('param.txt','w')
    cv_rs = pd.Series(cv_results).to_json(orient='split')
    print(json.dumps(cv_rs), file=grid_search_file)

def train(train_data, test_data):
    '''
    训练模型与评测
    调参得到最好的参数是：
    Best score: 0.984
    Best parameters set:
        cls__alpha: 0.6
        tfidf__norm: 'l2'
        tfidf__sublinear_tf: True
        tfidf__use_idf: True
        vect__max_df: 0.5
        vect__ngram_range: (1, 2)

    应用改模型，将模型持久化，并重新加载完成测试
    '''  
    data = [i[0] for i in train_data]
    label = [i[1] for i in train_data]
    pipline = Pipeline([
        ('vect', CountVectorizer(min_df=1, max_df=0.5, ngram_range=(1,2),decode_error = 'ignore', analyzer='char')),
        ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True, use_idf=True))
    ])

    label = np.array(label)
    text_fea = pipline.fit_transform(data)
    joblib.dump(pipline, TF_IDF_MODEL)
    clf = MultinomialNB(alpha=0.6)
    clf.fit(text_fea, label)
    joblib.dump(clf, NB_MODEL)

    # 测试在测试集上的表现
    p = joblib.load(TF_IDF_MODEL)
    clf2 = joblib.load(NB_MODEL)
    X_test = [i[0] for i in test_data]
    y_test = [i[1] for i in test_data]
    
    t1 = time.time()
    X_fea = p.transform(X_test)
    y_pred = clf2.predict(X_fea)
    t2 = time.time()
    print("total: ", len(y_pred), "right: ",
          sum(y_pred==y_test), 'accuracy:', sum(y_pred==y_test) / len(y_pred))
    print("f1:score", f1_score(y_test, y_pred, average='binary'))
    print(classification_report(y_test, y_pred, digits=5))
    print(confusion_matrix(y_test, y_pred))
    print(t2 - t1, 'sec!')


class NBModel():
    '''
    简单封装，以供调用
    '''
    def __init__(self):
        self.p = joblib.load(TF_IDF_MODEL)
        self.clf = joblib.load(NB_MODEL)

    def predict(self, sms):
        fea = self.p.transform([sms])
        return self.clf.predict(fea)

    def predict_prob(self, sms):
        fea = self.p.transform([sms])
        return self.clf.predict_proba(fea)



def main():
    print('data reading ...')
    # train_data = load_data(TRAIN_DATA)
    # test_data = load_data(TEST_DATA)
    # alchemist_train(train_data, test_data)

    train_data = load_data(TRAIN_DATA)
    test_data = load_data(TEST_DATA)
    print('train_data_length:', len(train_data))
    print('example 1:', train_data[0])
    train(train_data, test_data)




if __name__ == '__main__':
    main()