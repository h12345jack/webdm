# -*- coding: utf-8 -*-
import jieba
import jieba.posseg as pseg
import sklearn
import time as t
from data_import import load_data
from sklearn.externals import joblib
from scipy import sparse, io
import json

#重写TfIdfVectorizer类
class TfidfVectorizer(sklearn.feature_extraction.text.TfidfVectorizer):
    def build_analyzer(self):
        def analyzer(doc):
            #words = pseg.cut(doc)
            #new_doc = ''.join(w.word for w in words if w.flag != 'x')
            #new_doc = ''.join(w.word for w in words if (w.flag != 'x') and (w.flag != 'eng'))
            words = jieba.cut(doc)
            return words
        return analyzer

#创建词向量
def create_word_vec():
    t0 = t.time()
    training_data, training_label = load_data(sys.path[0].replace('\\','/')+'/datas/training_data.txt')
    print(len(training_data))

    vec_tfidf = TfidfVectorizer(min_df=2, max_df=0.8)
    training_data = vec_tfidf.fit_transform(training_data)
    joblib.dump(vec_tfidf,sys.path[0].replace('\\','/')+'/model/tf_idf.pkl')

    io.mmwrite(sys.path[0].replace('\\','/')+'/datas/word_vector_train.mtx', training_data)
    with open(sys.path[0].replace('\\','/')+'/datas/train_label.json', 'w') as f:
        json.dump(training_label, f)
    print("compute tfidf: %0.3fs" % (t.time() - t0))
    return training_data, training_label
