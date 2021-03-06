# -*- coding: utf-8 -*-
import pandas as pd

ks = 10

all_models =[{'type':'CNN', 'input':'onehot', 'pretrained':False, 'name':'CNN_onehot'},
              {'type':'CNN', 'input':'onehot', 'pretrained':True, 'name':'CNN_onehot_pretrained'},
              {'type':'CNN', 'input':'matrix', 'pretrained':False, 'name':'CNN_matrix'},
              {'type':'CNN', 'input':'matrix', 'pretrained':True, 'name':'CNN_matrix_pretrained'},
              {'type':'LSTM', 'input':'onehot', 'pretrained':False, 'name':'LSTM_onehot'},
              {'type':'LSTM', 'input':'onehot', 'pretrained':True, 'name':'LSTM_onehot_pretrained'}]



all_model_ks = {model['name'] + '_' + str(k):model['name'] for model in all_models for k in range(ks) }



model_names = [x['name'] for x in all_models]
models_k = [x+'_'+str(k) for x in model_names for k in range(ks)]



CNNs = [x['name'] for x in all_models if x['type'] == 'CNN']


reverse_tab = str.maketrans("ACUG", "UGAC")
def reverse_complement(seq):
    return seq.translate(reverse_tab)[::-1]



sections = ['pad_left', 'A_tail', 'stem_left', 'loop', 
            'stem_right', 'U_tail', 'pad_right']

  

seq_length={'tRNAs':95, 'terminators':75}



def check_and_make_dir(*mydirs):
    import os
    for mydir in mydirs:
        if not os.path.exists(mydir):
            os.makedirs(mydir)
            print('created directory', mydir)



def read_arnold_validation(INF):
    with open (INF, 'r') as inf:
        arnold = inf.read()
    arnold = [x for x in arnold.split('>') if x]
    found = [0 if x.find('No predicted transcription') != -1 else 1 for x in arnold]
    ids = [x.split('\n')[0] for x in arnold]
    ids_split = [x.split('_') for x in ids]
    data = pd.DataFrame({'id':ids,
                         'idx':[int(x[0]) for x in ids_split],
                         'k':[int(x[1]) for x in ids_split],
                         'y_true':[int(x[2]) for x in ids_split],
                         'y_pred': found})
    return data
