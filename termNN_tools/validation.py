# -*- coding: utf-8 -*-

import pandas as pd


def calc_validation(y_true, y_pred):
    
    TP = ((y_true==1)&(y_pred==1)).sum()
    FP = ((y_true==0)&(y_pred==1)).sum()
    TN = ((y_true==0)&(y_pred==0)).sum()
    FN = ((y_true==1)&(y_pred==0)).sum()
    
    P = TP+FN
    N = TN+FP
    
    accuracy = (TP+TN)/(P+N)
    precision = TP/(TP+FP)
    recall = TP/P
    selectivity = TN/N
    
    return {'acc':accuracy, 'prec':precision, 'rec':recall, 'select':selectivity}



def read_arnold(INF):
    with open (INF, 'r') as inf:
        arnold = inf.read()
    arnold = [x for x in arnold.split('>') if x]
    found = [0 if x.find('No predicted transcription') != -1 else 1 for x in arnold]
    ids = [x.split('\n')[0] for x in arnold]
    ids_split = [x.split('_') for x in ids]
    data = pd.DataFrame({'id':ids,
                         'idx':[int(x[0]) for x in ids_split],
                         'k':[int(x[1]) for x in ids_split],
                         'type':[int(x[2]) for x in ids_split],
                         'value': found})
    return data




