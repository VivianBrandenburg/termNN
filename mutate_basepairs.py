# -*- coding: utf-8 -*-

import termNN_tools.mutations as mut 
from termNN_tools.encoding import choose_predict_method
from termNN_tools.toolbox import all_models

from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np



for RNA_type in ['tRNAs', 'terminators']:

    
    # files and dirs 
    PATH_models = f'models/{RNA_type}/'
    PATH_validation_data = f'data/{RNA_type}/training/'
    INF_struc = f'data/{RNA_type}/mutation/{RNA_type}_structures.csv'
    FILE_bp_results = f'results/{RNA_type}/mutation/bp_results.csv'
    FILE_bp_seqs = f'results/{RNA_type}/mutation/bp_seqs.csv'
    
    
    # other vars
    ks = 10
    repeats = 15 
    if RNA_type == 'tRNAs': max_mutated_bp, merge_key = 20, 'unpadded_seq'
    if RNA_type == 'terminators': max_mutated_bp, merge_key = 9, 'sequence'
    
    
    # safe results
    seqs_bp = pd.DataFrame()
    results_bp = pd.DataFrame(columns = ['model', 'k', 'mutation', 'kept', 'lost'])
    
    
    for k in range(ks):
        
        # make testdata
        INF_valdata = f'{PATH_validation_data}k{str(k)}/val.csv'
        data = mut.get_testdata(INF_valdata, INF_struc, merge_key)
    
        
        # =============================================================================
        # mutate 
        # =============================================================================
    
        print(f'mutating k={k}')
        seqs_kept = mut.mutate_basepairs(data, repeats, 'keep', max_mutated_bp)
        seqs_lost = mut.mutate_basepairs(data, repeats, 'loose', max_mutated_bp)
        
        for key in seqs_kept.keys():
            seqs_bp = seqs_bp.append([pd.DataFrame({'k':k, 'type':'kept','mutation':key, 'seqs':seqs_kept[key]}),
                                      pd.DataFrame({'k':k, 'type':'lost','mutation':key, 'seqs':seqs_lost[key]})],
                                     ignore_index=True)
     
        # =============================================================================
        # load models
        # =============================================================================
        for m in all_models:
            
            # model selection
            model_name = m['name']
         
            # load model
            INF_model = f'{PATH_models}{model_name}/{model_name}_k{str(k)}.h5'
            model = load_model(INF_model)
            predict = choose_predict_method(model_name)
            
                
            # =============================================================================
            # mutate basepairs 
            # =============================================================================
                   
            # do mutations modus='keep'
            print(f'modeltype={model_name}, k={k}, type=keep')
            kept= {key:predict(model, np.array(value)) for key,value in seqs_kept.items()}
            kept= {key:np.mean(value) for key,value in kept.items()}
            
            # do mutations modus='loose'
            print(f'modeltype={model_name}, k={k}, type=loose')
            lost = {key:predict(model, np.array(value)) for key,value in seqs_lost.items()}
            lost = {key:np.mean(value) for key,value in lost.items()}
    
            # safe predictions
            for key in kept.keys():
                results_bp = results_bp.append({'model':model_name, 'k':k, 'mutation':key, 'kept':kept[key], 'lost':lost[key]}, ignore_index=True)
    
    
              
        # =============================================================================
        # write out results 
        # =============================================================================
        
    results_bp.to_csv(FILE_bp_results, index=False)  
    
    seqs_bp['id'] = ['_'.join([str(a), str(b), str(c), str(d)]) for a,b,c,d in 
                     zip(seqs_bp.index, seqs_bp.k, seqs_bp.type, seqs_bp.mutations)]
    seqs_bp.to_csv(FILE_bp_seqs, index=False)  
     




