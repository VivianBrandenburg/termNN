#!/usr/bin/env python3


import pandas as pd
from termNN_tools.plots import plot_sendseq
from termNN_tools import validation as v
from sklearn.metrics import auc
from termNN_tools.toolbox import all_model_ks
import os


INF_fna = "data/terminators/ecoli_genome/GCF_000005845.2_ASM584v2_genomic.fna"
INF_TTS = "data/SEndSeq/ju2019_suplTable2_TTS.csv"
INF_TU = "data/SEndSeq/ju2019_suplTable3_TU.csv"
INF_arn = 'data/arnold/genomescan_ecoli_processed.csv'

DIR_scan = 'results/terminators/genomescan/results_single_nts/'
DIR_kernels = 'results/terminators/genomescan/transcriptome_kernels/'
DIR_plots = 'results/terminators/genomescan/'

strands = [['-', 'rev'], ['+', 'fwd']]
max_dists = [10,15,35,50,100, 150, 250]


# =============================================================================
#  read in ju data
# =============================================================================

fna = v.read_fna(INF_fna, convert_TU=True)
tts_all = pd.read_csv(INF_TTS, comment='#')
tts_all.TTS_position = tts_all.TTS_position.astype(int)
tu_all = pd.read_csv(INF_TU, comment='#')


# =============================================================================
# build kernels
# =============================================================================

build_kernels = False

if build_kernels:
    for strand in strands:
        tu = tu_all[tu_all.strand == strand[0]]
        transcriptome = v.transcriptome_ju2019(tu, 150, fna)
        
        # read in genomescan results
        files_scan = os.listdir(DIR_scan+strand[1]+'/')
        scan = [pd.read_csv(DIR_scan+strand[1]+'/' + x, index_col=0) for x in files_scan]
        scan = pd.concat(scan, ignore_index=True)
        
        # filter scan for expressed regions
        t_scan = pd.DataFrame()
        for i in transcriptome.index:
            t_scan = t_scan.append(scan[(scan.nt_ref >= transcriptome.loc[i].start) &
                                        (scan.nt_ref <= transcriptome.loc[i].stop)],
                                   ignore_index=True)
        
        # build kernels
        for model_k in all_model_ks.keys():
            print('finding kernels for', model_k, strand[1])
            scores = t_scan[[model_k, 'nt_ref']]
            scores = scores[scores[model_k] >= 0.5]
            fused = v.find_kernels(scores, model_k)
            fused.to_csv(DIR_kernels + strand[1] + '/' +  model_k + '.csv', index=False)
        


# =============================================================================
# compare kernels with SEnd-Seq
# =============================================================================
    

res = pd.DataFrame()
for strand in strands:
 
    # make strand-specific
    tts = tts_all[tts_all.TTS_strand == strand[0]]
    tu = tu_all[tu_all.strand == strand[0]]
    transcriptome = v.transcriptome_ju2019(tu, 150, fna)
    
    # tts outside of expressed regions
    orphans = v.identify_orphans(tts, transcriptome, 'TTS_position')

    
    for model_k, model in all_model_ks.items():
        # read in data
        fused = pd.read_csv(DIR_kernels + strand[1] + '/' +  model_k + '.csv')
        
        # add ditance to next tts
        fused['next_TTS'], fused['dist' ]= v.get_distance(fused.center, [int(x) for x in tts.TTS_position])
        
        # add false negatives (tts which have no next kernel)
        fused = fused.append(v.false_negatives(fused.next_TTS, tts, orphans),
                              ignore_index=True)
        
        # add some info
        fused[['model', 'model_k', 'strand']] = model, model_k, strand[0]
        
        res = res.append(fused)


# =============================================================================
#  calc arnold 
# =============================================================================
    
    # prepare data
    arn_all = pd.read_csv(INF_arn, index_col=0)
    arn_all = arn_all[arn_all.strand==strand[0]]
    arn_all['center'] = [x + int(len(y)/2) for x,y in 
                         zip(arn_all.start, arn_all.term_sequence)]
    arn_all = arn_all[['start', 'center']]
   
    # exclude non-transcriptome hits
    arn = pd.DataFrame()
    for i in transcriptome.index:
        arn_i = arn_all[(arn_all.center >= transcriptome.loc[i].start) &
                        (arn_all.center <= transcriptome.loc[i].stop)]
        
        arn = arn.append(arn_i, ignore_index=True)

    # find next SEnd-Seq hit
    arn = arn.sort_values(by='center').reset_index(drop=True)
    arn['next_TTS'], arn['dist'] = v.get_distance(arn.center, tts.TTS_position)
    arn['max_score'] = 1
     
    # add false negatives
    arn = arn.append(v.false_negatives(arn.next_TTS, tts, orphans),
                            ignore_index=True)
    arn[['model', 'model_k', 'strand']] = 'arnold', 'arnold', strand[0]
    res = res.append(arn, ignore_index = True)


# =============================================================================
# calc precision-recall curve
# =============================================================================
prerec=pd.DataFrame()  
for dist in max_dists:
    for model_k, model in  all_model_ks.items() :
        data_s = res[res.model_k == model_k].sort_values(by='center')
        prerec_s = v.prerec_termNN(data_s, dist)
        prerec_s[['model', 'model_k', 'max_dist']] = model, model_k, dist
        prerec = prerec.append(prerec_s)
    
    # same for arnold
    arn_prerec = v.prerec_arnold(res[res.model == 'arnold'].reset_index(drop=True), dist)
    arn_prerec[['model', 'model_k', 'max_dist']] = 'arnold', 'arnold', dist
    prerec = prerec.append(arn_prerec)
                           


# =============================================================================
#  calc area under precision-recall curve
# =============================================================================

aucs = pd.DataFrame()

for model_k, model in  all_model_ks.items():
    data_k = prerec[prerec.model_k == model_k]
    for dist in max_dists:
        data_d = data_k[data_k.max_dist == dist] 
        auc_d= {'model':model,'model_k':model_k, 
                'auc':auc(data_d.rec,  data_d.prec),
                'max_dist':dist,'strand':strand[0]}
        aucs = aucs.append(auc_d, ignore_index=True)
   
       

# =============================================================================
# plot 
# =============================================================================

plot_sendseq(aucs, prerec, 'CNN', DIR_plots)
plot_sendseq(aucs, prerec, 'LSTM', DIR_plots)
