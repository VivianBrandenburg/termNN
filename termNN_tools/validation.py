# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import precision_score, recall_score
from itertools import cycle


import copy


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


# =============================================================================
# 
# =============================================================================


def TU_converter(seq):
    if seq[:1000].find('U') == -1:
        return seq.replace('T', 'U')
    if seq[:1000].find('T') == -1:
        return seq.replace('U', 'T')





def read_fna(INF, convert_TU=False):
    with open(INF) as inf:
        fna = inf.readlines()[1:]
        fna = [x.strip() for x in fna]
        fna = ''.join(fna)
        if convert_TU:
            return TU_converter(fna)
    return fna




def transcriptome_ju2019(data, elongate, fna):  
    len_fna = len(fna)
    borders = [list(range(x-elongate,y+elongate,1)) for x,y in zip(data.start, data.stop)]
    borders = sorted(list(set([i for x in borders for i in x])))
    
    # fuse overlapping transcripts
    start, stop = [], []
    for i,n in enumerate(borders[:-1]):
        if n!= borders[i-1]+1:  start.append(n)
        if n!= borders[i+1]-1:  stop.append(n)
    stop.append(borders[-1])
    
    # cut nts at genome borders
    start = [max(x,0) for x in start] # no negative starts
    stop = [min(x,len_fna) for x in stop] # no stops after genome end
    return pd.DataFrame({'start': start, 'stop': stop})



def identify_orphans(tts, exp, select_col = 'match_center'):
    tts_outside_of_tu = pd.DataFrame()
    for i in tts.index:
        test = exp[(exp.start < tts.loc[i][select_col]) & 
                         (exp.stop > tts.loc[i][select_col])]
        if len(test) ==0:
            tts_outside_of_tu = tts_outside_of_tu.append(tts.loc[i])
    return tts_outside_of_tu





def fuse(scores, start, stop, model_k):
    max_score = max(list(scores.loc[start:stop][model_k]))
    a = scores.loc[start].nt_ref
    b = scores.loc[stop].nt_ref
    return {'start':a, 'stop':b, 'center':int((a+1+b)/2), 
            'length': b-a+1,
            'max_score':max_score}


def find_kernels(scores, model_k):
    fused = pd.DataFrame()
    indices = scores.index
    
    start, stop = -99, -99
    for n, i in enumerate(indices):
        if n == 0:
            start, stop = i,i 
        else:
            if i == indices[n-1] +1:
                stop = i
            else:
                fused = fused.append(fuse(scores, start, stop, model_k), ignore_index=True)
                start = i
                stop = i
            if n == len(indices)-1:
                fused = fused.append(fuse(scores, start, stop, model_k), ignore_index=True)
    return fused

     





def get_distance(kernel_pos, tts_pos):
    tts_pos = sorted(tts_pos)
    cycle_gff = cycle(tts_pos)
    closest = next(cycle_gff) 
    closest2 = next(cycle_gff)
    next_tss, dist_next = [], []
    for x in kernel_pos:
        while abs(x-closest) > abs(x-closest2):
            closest = copy.deepcopy(closest2)
            closest2 = next(cycle_gff)
        next_tss.append(closest)
        dist_next.append(x-closest)
    return next_tss, dist_next


# add false negatives
def false_negatives(positives, tts, orphans):
    fn = tts[~ tts.TTS_position.isin(list( orphans.TTS_position))]
    fn = fn[~fn.TTS_position.isin(list(positives))]
    return  pd.DataFrame({'dist':0, 'max_score':0, 'next_TTS':fn.TTS_position})
    



# =============================================================================
# precision recall curves 
# =============================================================================


from sklearn.metrics import precision_recall_curve

def prerec_termNN(data, dist):
    
    y_true = data.dist.abs() <= dist
    y_predict = data.max_score

    prec, rec, thresh = precision_recall_curve(y_true=y_true, probas_pred=y_predict)
    res_d = pd.DataFrame({'prec': prec, 'rec':rec})
    
    resA = res_d.loc[0:40]
    resB = res_d.loc[40:len(res_d)-40].sample(frac=0.01)
    resB = resB.sort_index().reset_index(drop=True)
    resC = res_d.loc[len(res_d)-40:len(res_d)]
    res_d = pd.concat([resA, resB, resC])
    
    return pd.DataFrame({'prec': res_d.prec, 'rec':res_d.rec })
    




# precision and recall for arnold
def prerec_arnold(arnold, dist):
    arnold['y_true'] = arnold.dist.abs() <= dist 
    prec = precision_score(y_true = arnold.y_true, y_pred = arnold.max_score)
    rec = recall_score(y_true = arnold.y_true, y_pred = arnold.max_score)
    return pd.DataFrame({'prec': [prec], 'rec': rec})
     
     
        
