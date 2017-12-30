# -*- coding: utf-8 -*-
import Logistic_Predictor as LRP
from data_import import load_data
import time as t
import os, sys

if '__main__' == __name__:
    t0 = t.time()
    #test
    print("start predict:")
    test_data, test_label = load_data(sys.path[0].replace('\\','/')+'/datas/test_data.txt')
    print(len(test_data))

    #predict data
    #参数为原始数据，test_data, test_label。
    predictor = LRP.Logistic_Predictor()
    
    #reset data
    result = predictor.predict_proba(test_data[11])
    #result 第一个元素大:0 ; 第二个元素大:1
    print(result, test_label[8])

    print("predicting time: %0.3fs" % (t.time() - t0))
