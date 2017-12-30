#coding=utf8
import time

from flask import Flask
from flask import jsonify
from flask_cors import CORS
from flask import request

from svm_spam_message.svm_train import get_trained_svm
from svm_spam_message.svm_predict import predict as svm_predict

from lr_spam_message.Logistic_Predictor import Logistic_Predictor

app = Flask(__name__)
CORS(app)

svm_model = get_trained_svm()
lr_model = Logistic_Predictor()

def svm_cls(message):
    sms = message
    is_spam = svm_predict(svm_model, sms)
    if is_spam:
        is_spam = 1
    else:
        is_spam = 0
    return {'prob': 1.0, 'is_spam': is_spam}

def lr_cls(message):
    result = lr_model.predict_proba(message)
    if result[0][0] > result[0][1]:
        is_spam = 0
        prob = result[0][0]
    else:
        is_spam = 1
        prob = result[0][1]

    return {'prob': prob, 'is_spam': is_spam}

def cls(message, model):
    if model == '1':
        #调用svm
        return svm_cls(message)
    if model == '0':
        # 调用LR
        return lr_cls(message)
    if model == '1':
        return {'prob': 1.0, 'is_spam': 0}
    return {'prob': prob, 'is_spam': is_spam}

@app.route('/api/message', methods = ['POST', 'GET'])
def message():
    if request.method == 'GET':
        return 'hello'
    if request.method == 'POST':
        req = request.json
        message = req.get('message')
        model = req.get('model', '1')
        return jsonify(cls(message, model))

if __name__ == '__main__':
    app.run(debug=True)