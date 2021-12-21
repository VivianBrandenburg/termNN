# -*- coding: utf-8 -*-

import pandas as pd


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
                         'type':[x[2] for x in ids_split],
                         'mutations':['_'.join(x[3:]) for x in ids_split],
                         'value': found})
    data = data.sort_values(by='idx').reset_index()
    return data




def calc_means(data, model_name, mut_types):
    print('calculating means for', model_name)
    data_means = pd.DataFrame(columns = ['model', 'k', 'mutations']+mut_types)
    for mutations in data.mutations.unique():
        for k in data.k.unique():
            df = data[(data.mutations == mutations) & (data.k == k)]
            
            res = {'model':model_name, 'k':k, 'mutations': mutations}
            for m_type in mut_types:
                res[m_type] = df[df.type == m_type].value.mean()
            
            data_means = data_means.append(res, ignore_index=True)
    return data_means
    
