# -*- coding: utf-8 -*-

from termNN_tools.encoding import onehot_encoding, matrix_encoding
from termNN_tools.toolbox import reverse_complement, all_models, check_and_make_dir, models_k

from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np

from itertools import cycle, groupby
import copy


from operator import itemgetter

import os


Max_length=75



    

# =============================================================================
# get genome slices
# =============================================================================



def get_gff(INF_gff):
    with open (INF_gff, 'r') as inf:
        gff = inf.read().strip().split('\n')
        gff = [x.split('\t') for x in gff if x[0] !='#']
        gff = [x[3] if x[6]=='-' else x[4] for x in gff]
        gff = [int(x) for x in gff]
    return sorted(set(gff))


def get_genome(INF_genome):
    with open (INF_genome, 'r') as inf:
        genome = inf.read()
        genome = ''.join([x for x in genome.strip().split('\n') if x[0] != '>'])
    return genome.replace('T','U')
    




def get_closest_CDS(ref_positions, gff):
    closest_CDS, dist_CDS = [], []
    
    cycle_gff = cycle(gff)
    closest = next(cycle_gff) 
    closest2 = next(cycle_gff)
    
    for x in ref_positions:
        while abs(x-closest) > abs(x-closest2):
            closest = copy.deepcopy(closest2)
            closest2 = next(cycle_gff)
        closest_CDS.append(closest)
        dist_CDS.append(x-closest)
    return closest_CDS, dist_CDS




# =============================================================================
#prep genome scan by encoding entire genome
# =============================================================================

def encode_genome(genome, gff, splits, DIR_out, reverse):
    # prep data
    slices = [genome[x:x+75] for x in range(0,len(genome)-75,3)]
    nt_ref = [x+38 for x in range(0,len(genome)-75,3)]
    next_CDS, dist_CDS = get_closest_CDS(nt_ref, gff)
    
    # reverse seqs if needed
    if reverse:
        print('reversing genomic sequences')
        slices = [reverse_complement(x) for x in slices]
    
    
    #write out data
    data = pd.DataFrame({'seq':slices, 'nt_ref': nt_ref, 
                          'next_CDS': next_CDS, 'dist_CDS': dist_CDS})
    data.to_csv(DIR_out+'genome_slices.csv')

    # prep slicing
    split_length = int(len(data)/splits)    
    start, stop = 0, split_length
    
    for i in range(splits):
        if stop>len(data):
            stop=len(data)
        
        split_of_slice = slices[start:stop]
        split_of_data = pd.DataFrame({'seq':slices[start:stop], 'nt_ref': nt_ref[start:stop], 
                         'next_CDS': next_CDS[start:stop], 'dist_CDS': dist_CDS[start:stop]})
        
        split_of_data.to_csv(DIR_out+ str(i) + '_genome.csv')
        slices_onehot = np.array([onehot_encoding(x) for x in split_of_slice])
        np.save(DIR_out +str(i)+ '_slices_onehot.npy', slices_onehot)
        print(f'prepared k={i}, onehot encoding')
        
        # slices_matrix = np.array([matrix_encoding(x) for x in split_of_slice])
        # np.save(DIR_out+ str(i)+ '_slices_matrix.npy', slices_matrix)
        # print(f'prepared k={i}, matrix encoding')
    
        start += split_length
        stop  += split_length 
        
     
# =============================================================================
# do genome scan 
# =============================================================================
        

def scan_genome(splits, ks, DIR_encodings, DIR_models, DIR_results):
    for s in range(splits): #splits 
        data = pd.read_csv(DIR_encodings + str(s)+'_genome.csv', index_col=0)
        
        for k in range(ks):
            
            for model_vars in all_models:
                
                # model selection
                input_type = model_vars['input']
                model_name = model_vars['name']
                
                model = load_model(DIR_models+model_name+'/'+model_name+'_k'+str(k)+'.h5')
                
                
                # get encoded input
                if input_type=='onehot':
                    seqs_encoded = np.load(DIR_encodings + str(s) + '_slices_onehot.npy')
                elif input_type=='matrix':
                    seqs_encoded = np.load(DIR_encodings + str(s) + 'slices_matrix.npy')
                
                data[model_name+'_'+str(k)] = model.predict(seqs_encoded)
                
                print(f'predicted slice={s}, k={k}, model={model_name}')
                
        data.to_csv(DIR_results +str(s)+'_results.csv')
            






# =============================================================================
# find kernels 
# =============================================================================



def get_ranges(data):
    ranges =[]    
    for k,g in groupby(enumerate(data),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        ranges.append((group[0],group[-1]))
    return ranges



def find_kernels_per_model(data, model, cutoff):
        model_k = data[data[model]>cutoff][data.columns.intersection([model, 'nt_ref', 'dist_CDS'])]
        
        ranges = get_ranges(model_k.index)
        res = pd.DataFrame({'index_range': ranges,
                                'frame_len': [(j-i+1)*3 for i,j in ranges],
                                'nt_ref':[[model_k.nt_ref[i], model_k.nt_ref[j]] for i,j in  ranges],
                                'dist_CDS':[min(model_k.loc[i:j].dist_CDS, key=abs) for i,j in ranges],
                                'prediction':[max(model_k.loc[i:j][model]) for i,j in ranges]})
        return res



        

def find_kernels(ks, cutoff, DIR_in, DIR_out, reverse):

    FILES = [x for x in os.listdir(DIR_in) if x.find('.csv') != -1]
    FILES.sort(key= lambda x: float(x.strip('_results.csv')))
    data = pd.DataFrame()
    for FILE in FILES:
        data = pd.concat([data, pd.read_csv(DIR_in+FILE, index_col=0)], ignore_index=True)
    
    for model_k in models_k:
        print(f'finding kernels in {model_k}')
        res = find_kernels_per_model(data, model_k, cutoff)        
        res.to_csv(DIR_out+model_k+'.csv')
    return res
        
        

# =============================================================================
# analyze genome scan, calc PPV     
# =============================================================================


# def calc_scan_PPVs(splits, max_dist_CDS, prediction_cutoff, DIR_results):
#     tp, fp = pd.DataFrame(), pd.DataFrame()
#     for s in range(splits):
#         print(s)
    
#         # real data
#         data_s = pd.read_csv(DIR_results + str(s) + '_results.csv', index_col=0)
#         res = data_s.drop(['seq', 'nt_ref', 'next_CDS', 'dist_CDS'], axis=1)
        
#         # data processed into y_true and y_predict
#         y_true = data_s.dist_CDS
#         y_true = abs(y_true) <= max_dist_CDS
#         y_pred = res > prediction_cutoff
        
#         #calc TP and FP
#         TP, FP = [], []
#         for col in y_pred.columns:
#             TP.append(((y_pred[col] == True) & (y_true == True)).sum())    
#             FP.append(((y_pred[col] == True) & (y_true == False)).sum()) 
        
#         #store TP and FP
#         tp = pd.concat([tp, pd.DataFrame(data=[TP], columns=res.columns)])
#         fp = pd.concat([fp, pd.DataFrame(data=[FP], columns=res.columns)])
        
        
#     res  = pd.DataFrame({'tp': tp.sum(), 'fp': fp.sum()})    
#     res['PPV'] = res.tp / (res.tp+res.fp)    
#     res['hits']=res.tp+res.fp
#     res['model_long'] = res.index
#     res['model']= [ '_'.join(x[:-1]) for x in res.model_long.str.split('_')]
#     res['k']= [ int(x[-1]) for x in res.model_long.str.split('_')]
    
#     return res    



def calc_PPV(predictions, dist_CDS, pdc_cut, dist_cut):
    tp = ((predictions >= pdc_cut) & (abs(dist_CDS) <= dist_cut)).sum()
    fp = ((predictions >= pdc_cut) & (abs(dist_CDS) > dist_cut)).sum()
    PPV = tp/(tp+fp)
    return PPV
    
    





# def sum_hits(splits, prediction_cutoff, DIR_results):
#     dist_at_hits = pd.DataFrame()  
#     for s in range(splits):
#         print(s)
#         data_s = pd.read_csv(DIR_results + str(s) + '_results.csv', index_col=0)
        
#         # separate data and non-data columns
#         non_data_names = ['seq', 'nt_ref', 'next_CDS', 'dist_CDS']
        
#         data_only = data_s.drop(non_data_names, axis=1 )
#         dist_CDS = pd.DataFrame({key:data_s.dist_CDS for key in data_only.columns})
        
#         # filter data for hits
#         hits_mask = data_only > prediction_cutoff
#         dist_at_hits_s = dist_CDS[hits_mask==True].dropna(axis=0, how='all').dropna(axis=1, how='all')
#         dist_at_hits = pd.concat([dist_at_hits, dist_at_hits_s])
        
#     return dist_at_hits
        
        



def get_best_hits(DIR_in, arnold_data, strand):
    res = pd.DataFrame()
    for model_k in models_k:
        selected = pd.read_csv(DIR_in + model_k + '.csv', index_col=0)
        selected = selected.sort_values(by='prediction', ascending=False).reset_index()
        selected = selected[:len(arnold_data)]
        selected['strand'] = strand
        selected['model_k'] = model_k
        selected['model'] = '_'.join(model_k.split('_')[:-1])
        res = pd.concat([res, selected], ignore_index=True)
    arnold_data['model']= arnold_data['model_k']='ARNold'
    res = pd.concat([res, arnold_data], ignore_index=True)
    return res

    


def calc_PPV_best_kernels(DIR_kernels, FILE_arnold):
    # prep arnold data    
    arnold = pd.read_csv(FILE_arnold, index_col=0)
    arnold = pd.DataFrame({'model': 'ARNold', 'model_k': 'ARNold',
                           'index_range':0, 'frame_len': arnold.length,
                           'nt_ref': arnold.center, 'dist_CDS':arnold.dist_CDS,
                           'strand':arnold.strand, 'prediction':1 })
    arnold_fwd = arnold[arnold.strand=='+'].reset_index()
    arnold_rev = arnold[arnold.strand=='-'].reset_index()
    
    # find best hits 
    fwd = get_best_hits(DIR_kernels+'fwd/', arnold_fwd, '+')
    rev = get_best_hits(DIR_kernels+'rev/', arnold_rev, '-')
    best_hits = pd.concat([fwd, rev], ignore_index=True)
    
    # calc PPVs 
    prediction_cut, distanceCDS_cut = 0.5, 250
    res = pd.DataFrame()
    for m in best_hits.model_k.unique():
        df = best_hits[best_hits.model_k == m]
        PPV = calc_PPV(df.prediction, df.dist_CDS, prediction_cut, distanceCDS_cut)
        res = res.append({'model': df.iloc[0].model, 'model_k':m, 'PPV':PPV}, ignore_index=True)
    return res
    
    
    # # =============================================================================
    # # plot 
    # # =============================================================================
    
    # plot_genomescan(res, 'genomescan, kernels, best hits, all', res.model.unique() )
    # plot_genomescan(res, 'genomescan, kernels, best hits, CNN', [x for x in res.model.unique() if x.find('LSTM') == -1] )
    # plot_genomescan(res, 'genomescan, kernels, best hits, LSTM', [x for x in res.model.unique() if x.find('LSTM') != -1] )
    


    