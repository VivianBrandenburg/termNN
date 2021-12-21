# -*- coding: utf-8 -*-

import random
import pandas as pd
import copy
from termNN_tools.toolbox import all_models
from tensorflow.keras.models import load_model
from termNN_tools.encoding import choose_predict_method
import numpy as np


# =============================================================================
# get data for mutations
# =============================================================================


def get_testdata(INF_test_data, INF_struc_data, merge_key):
    test_data = pd.read_csv(INF_test_data, index_col=0)
    test_data = test_data[test_data.label == 1]
    struc_data = pd.read_csv(INF_struc_data, index_col=0)
    data = pd.merge(test_data, struc_data, on=merge_key)
    return data



def find_pairings(structure):
    istart = []  # stack of indices of opening parentheses
    d = {}
    for i, c in enumerate(structure):
        if c == '(':
             istart.append(i)
        if c == ')':
            try:
                d[istart.pop()] = i
            except IndexError:
                print('Too many closing parentheses')
    if istart:  # check if stack is empty afterwards
        print('Too many opening parentheses')
    return d





## =============================================================================
## mutate basepairs
## =============================================================================

def mutate_keeping_bp(nt1, nt2):
    bp= [['G', 'C'],  ['C', 'G'],
         ['U', 'A'], ['A', 'U']]
    other_bp = [x for x in bp if x[0] != nt1 and x[1] != nt2]
    return random.choice(other_bp)


    
def mutate_loosing_bp(nt1, nt2):
    non_bp = [['G', 'A'], ['A', 'G'], 
              ['C', 'A'], ['A', 'C'],
              ['C', 'U'], ['U', 'C']]
    other_bp = [x for x in non_bp if x[0] != nt1 and x[1] != nt2]
    return random.choice(other_bp)
    


def mutate_basepairs(data, repeats, mode, max_mutated_bp):
    mutated_sequences = {key:[] for key in range(max_mutated_bp)}
    data = data.reset_index(drop=True)  
    
    for row in data.index:
        unpadded = data.loc[row].unpadded_seq # get right input
        padded = data.loc[row].sequence
        structure = data.loc[row].struc
        
        mutated_sequences[0].append(padded)
        basepairs = find_pairings(structure)
        basepairs = [[key, value] for key, value in basepairs.items()]
        
        max_mutated_bp_here = min(max_mutated_bp, len(basepairs))
        for number_of_mutations in range(1, max_mutated_bp_here, 1): 
            for repeat in range(1, repeats+1, 1): 
                bp_to_mutate = random.sample(basepairs, k=number_of_mutations) 
                new_unpadded = mutate_one_sequence(unpadded, bp_to_mutate, mode)
                new_padded = padded.replace(unpadded, new_unpadded) 
                mutated_sequences[number_of_mutations].append(new_padded)
        
    return mutated_sequences
        
        

def mutate_one_sequence(unpadded, bp_to_mutate, mode):
    mutated_seq = list(unpadded)
    for pos_bp in bp_to_mutate:
        pos_nt1, pos_nt2 = pos_bp
        old_nt1, old_nt2 = mutated_seq[pos_nt1], mutated_seq[pos_nt2]
        new_nt1, new_nt2 = mutate_one_pair(old_nt1, old_nt2, mode)
        mutated_seq[pos_nt1] = new_nt1
        mutated_seq[pos_nt2] = new_nt2
    return ''.join(mutated_seq)
            





def mutate_one_pair(i,j, mode):
    if mode == 'keep':
        new_pair = mutate_keeping_bp(i,j)
    elif mode == 'loose':
        new_pair = mutate_loosing_bp(i,j)
    else:
        raise KeyError("you've tried to use a mutation mode which doesn't exist")
    return new_pair




# =============================================================================
# make section mutations 
# =============================================================================

# mutation rules
fusion_order = ['pad_left', 'stretch_left', 'stem_left', 'loop',
                'stem_right', 'strech_right', 'pad_right']


def mutate_single_nt(nt):
    mutate={'A':['C','U', 'G'], 'U':['A','G', 'C'], 
            'C':['A','G', 'U'], 'G':['C','U', 'A']}
    return random.choice(mutate[nt])

    


def mutate_half_of_nts(seq):
    half_of_nts = int(len(seq)/2)
    if len(seq)%2 != 0: half_of_nts += random.choice([0,1])
    mutated_index = random.sample(range(len(seq)), k=half_of_nts)
    seq = [nt if i not in mutated_index else mutate_single_nt(nt) for i,nt in enumerate(seq)]
    return ''.join(seq)


def mutate_one_section(data, section_name, mutation_repeats):
    all_mutated_seqs = []
    
    for i in data.index:
        seq = {key: data.loc[i][key] for key in fusion_order}
        section = seq[section_name]
        
        for _ in range(mutation_repeats):
            mutated_section = mutate_half_of_nts(section)
            mutated_seq = copy.deepcopy(seq)
            mutated_seq[section_name]=mutated_section
            mutated_seq = ''.join([mutated_seq[key] for key in fusion_order])
            all_mutated_seqs.append(mutated_seq)
            
    return all_mutated_seqs


def predict_mutations(data, PATH_models, k):
    res = pd.DataFrame()
    for m in all_models:
        print('predicting', data.loc[0].type, 'for', m['name'], ', k =', k)
        
        # load model
        INF_model = f'{PATH_models}{m["name"]}/{m["name"]}_k{str(k)}.h5'
        model = load_model(INF_model)
        predict_method = choose_predict_method(m['input'])
        
        #predict per section
        for section in data.mutation.unique():
            data_part = data[data.mutation == section]
            prediction = predict_method(model, data_part.seq)
                 
            res = res.append({'model': m['name'], 'k':k, 
                              'mutation': section, 
                              'section': np.mean(prediction)}, ignore_index=True)
    return res       
        



    