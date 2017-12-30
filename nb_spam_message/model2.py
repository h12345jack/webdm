#coding=utf8
import os

import jieba

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

TRAIN_DATA = './data/training_data.txt'
TEST_DATA = './data/test_data.txt'
TRAIN_DATA_CUT = './data/training_data_wc.txt'

def read_train():
    train_data = []
    with open(TRAIN_DATA, encoding='utf8') as f:
        for line in f.readlines():
            line.strip()
            label_and_data = line.split('\t')
            label, data = label_and_data
            train_data.append((data, int(label)))
    return train_data


def model(train_data):
    '''
    tf_idf的特征, naive bayes
    '''  
    counter_vec = CountVectorizer()
    data = [i[0] for i in train_data]
    label = [i[1] for i in train_data]
    pipline = Pipeline([
        ('vect', CountVectorizer(min_df=1, decode_error = 'ignore', analyzer='char')),
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
        print(classification_report(y_test_folders, y_pred))

    


# # tfidf_pipline = Pipeline([
# #     ('tfidf', )

# # ])




print('data reading ...')

train_data = read_train()
print('train_data_length:', len(train_data))
# def main():
#     pass

print('example 1:', train_data[0])
model(train_data)



# if __name__ == '__main__':
#     main()