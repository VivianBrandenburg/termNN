# -*- coding: utf-8 -*-

import termNN_tools.mutations as mut
import pandas as pd



RNA_type = 'terminators'
ks=10
mut_repeats =15
    

# files and dirs 
PATH_models = f'models/{RNA_type}/'
PATH_validation_data = f'data/{RNA_type}/training/'
INF_struc = f'data/{RNA_type}/mutation/{RNA_type}_structures.csv'
FILE_results = f'results/{RNA_type}/mutation/sec_results.csv'
DIR_out = f'results/{RNA_type}/mutation/'


seqs_sec = pd.DataFrame()
results_sec = pd.DataFrame()
    
for k in range(ks):
    print('starting section mutations for k =', k)
    
    # make testdata
    INF_valdata = f'{PATH_validation_data}k{str(k)}/val.csv'
    data = mut.get_testdata(INF_valdata, INF_struc, 'sequence')

    # mutate sections
    sequences = pd.DataFrame({'k': k, 'type': 'section', 'mutation':'base',
                              'seq':list(data.padded_seq)})
    for section in mut.fusion_order:
        seq = mut.mutate_one_section(data, section, mut_repeats)
        sequences = sequences.append(pd.DataFrame({'k': k, 'type': 'section', 'mutation':section,                                       'seq':seq}), ignore_index=True)

    # predict 
    res = mut.predict_mutations(sequences, PATH_models, k)
    results_sec = results_sec.append(res)

    # save results 
    seqs_sec = seqs_sec.append(sequences)
              
                       
# write outfiles mutated sections
seqs_sec['id'] = ['_'.join([str(a), str(b), c,str(d)]) for a,b,c,d in
                   zip(seqs_sec.index,seqs_sec.k,seqs_sec.type,seqs_sec.mutation)]    
seqs_sec.to_csv(DIR_out + 'sec_seqsTRIAL.csv')
results_sec.to_csv(DIR_out + 'sec_resTRIAL.csv')
     









