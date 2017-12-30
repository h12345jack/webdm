#coding=utf8
import os

import jieba


from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


TRAIN_DATA = './data/training_data.txt'
TEST_DATA = './data/test_data.txt'
TRAIN_DATA_CUT = './data/training_data_wc.txt'

def word_cut(text):
    text = text.strip()
    w_list = [i for i in jieba.cut(text)]
    return w_list

def read_train():
    train_data = []
    if not os.path.exists(TRAIN_DATA_CUT):
        f2 = open(TRAIN_DATA_CUT, 'w', encoding='utf8')
        with open(TRAIN_DATA, encoding='utf8') as f:
            for line in f.readlines():
                line.strip()
                label_and_data = line.split('\t')
                assert len(label_and_data) == 2
                label, data = label_and_data
                data = word_cut(data)
                print(label, " ".join([w for w in data]), sep='\t', file=f2)
                train_data.append((data, label))
        f2.close()
    else:
        with open(TRAIN_DATA_CUT, encoding='utf8') as f:
            for line in f.readlines():
                label, data = line.split('\t')
                # data = data.split(" ")
                train_data.append((data.strip(), label))
    return train_data


def model1(train_data):
    '''
    tf_idf的特征
    '''        
    pipline = Pipeline([
        ('counter', CountVectorizer(decode_error = 'ignore')),
        ('tfidf', TfidfVectorizer())
    ])

    data = [i[0] for i in train_data]
    label = [i[1] for i in train_data]

    # count_vec = CountVectorizer()
    text_fea = pipline.fit_transform(data)

    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200,
                         random_state = 42)
    
    kfolder = KFold(n_splits=5, random_state=42)
    for train_index, test_index in kfolder.split(text_fea, label):
        clone_clf = clone(bdt)
        x_train_folders = text_fea[train_index]
        y_train_folders = label[train_index]
        x_test_folders = text_fea[test_index]
        y_test_folders = text_fea[test_index]

        clone_clf.fit(x_train_folders, y_train_folders)
        y_pred = clone_clf.predict(x_test_folders)
        print("f1:score", f1_score(y_test_folders, y_pred, average='binary'))


    


# tfidf_pipline = Pipeline([
#     ('tfidf', )

# ])




print('data reading ...')

train_data = read_train()
print('train_data_length:', len(train_data))
# def main():
#     pass

print('example 1:', train_data[0])
model1(train_data)



# if __name__ == '__main__':
#     main()