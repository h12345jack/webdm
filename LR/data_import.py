# -*- coding: utf-8 -*-

from scipy import sparse, io
import json

def load_data(path):
    label = []
    lines = []
    content = []
    with open(path, encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            lines.append(line)
    for line in lines:
        message = line.split('\t')
        label.append(message[0])
        content.append(message[1])
    return content, label

def read_word_vec():
    training_data = io.mmread(sys.path[0].replace('\\','/')+'/datas/word_vector_train.mtx')
    with open('datas/train_label.json', 'r') as f:
        training_label = json.load(f)
    return training_data, training_label

