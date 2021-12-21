# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np






# =============================================================================
# choose right input mode
# =============================================================================
def choose_input(model_name):
    if model_name.find('onehot')!=-1: get_data = get_data_onehot
    elif model_name.find('matrix')!=-1: get_data = get_data_matrix
    else:  raise ValueError('your chosen input type does not exist')
    return get_data
        

def choose_predict_method(model_name):
    if model_name.find('onehot')!=-1: predict_method = predict_onehot
    elif model_name.find('matrix')!=-1: predict_method = predict_matrix
    else:  raise ValueError('your chosen input type does not exist')
    return predict_method
        


# =============================================================================
# get data for One Hot encoding
# =============================================================================
OneHot_encoder ={'A':np.array([0,0,0,1], np.float16),
                 'U':np.array([0,0,1,0], np.float16),
                 'G':np.array([0,1,0,0], np.float16),
                 'C':np.array([1,0,0,0], np.float16)}  

def get_data_onehot(data_infile):
    data = pd.read_csv(data_infile)
    seq = np.array([[OneHot_encoder[y] for y in x ] for x in list(data['sequence'])])
    label =  np.array([np.array(x)for x in list(data.label)])
    return  seq, label


def onehot_encoding(seq_in):
    return np.array([OneHot_encoder[y] for y in seq_in ])

# =============================================================================
# get data for matrix encoding 
# =============================================================================


matrix_rules = {'CG':1, 'AU':0.66, 'GU':0.33, 'GC':1, 'UA':0.66, 'UG':0.33,
                    'AA':0, 'GG':0, 'CC':0, 'UU':0, 'GA':0, 'AG':0, 'CA':0, 'AC':0, 'UC':0, 'CU':0 }    

def matrix_encoding(seq):
    seq_length = len(seq)
    seq_rev = seq[::-1]
    myTable = []
    for i in seq:
        row=[matrix_rules[''.join([i,j])] for j in seq_rev]
        myTable.append(row)
    return np.array(myTable).reshape(seq_length,seq_length,1)
    



def get_data_matrix(infile):
    infile=pd.read_csv(infile)
    data = [matrix_encoding(seq) for seq in infile.sequence]
    label = infile.label
    return np.array(data), np.array(label)




# =============================================================================
# predict with OneHot Input
# =============================================================================


def predict_onehot(model, sequences):
    OneHotSeq = np.array([[OneHot_encoder[y] for y in x ] for x in sequences])
    predictions =  model.predict(OneHotSeq)
    prediction_flat = [item for sublist in predictions for item in sublist]
    return prediction_flat


# =============================================================================
# predict with matrix input
# =============================================================================

def predict_matrix(model, sequences):
    sequences = [matrix_encoding(seq) for seq in sequences]
    sequences = np.array(sequences)
    predictions = model.predict(sequences)
    prediction_flat = [item for sublist in predictions for item in sublist]
    return prediction_flat


