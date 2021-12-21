# -*- coding: utf-8 -*-

import pandas as pd
from termNN_tools.mutations_processing import read_arnold, calc_means
from termNN_tools.plots import plot_mutations_bp , plot_mutations_sc


RNA_type = 'terminators'


# =============================================================================
# infiles 
# =============================================================================
INF_bp_arnold = 'data/arnold/mutate_bp_res.txt'
INF_sc_arnold = 'data/arnold/mutate_sec_res.txt'

INF_bp_termNN = f'results/{RNA_type}/mutation/bp_res.csv'
INF_sc_termNN = f'results/{RNA_type}/mutation/sec_res.csv'

OUTF_bp = f'results/{RNA_type}/mutation/bp_processed.csv'
OUTF_sc = f'results/{RNA_type}/mutation/sc_processed.csv'


# =============================================================================
# basepairs
# =============================================================================

# get data
bp = pd.read_csv(INF_bp_termNN, index_col=0)


if RNA_type=='terminators':
    arnold_bp = calc_means(read_arnold(INF_bp_arnold), 'ARNold', ['kept', 'lost'])
    bp = pd.concat([bp, arnold_bp], ignore_index=True).reset_index(drop=True)

# calc differences
bp['diff_abs'] = bp.kept-bp.lost
bp['diff_rel'] = 1-(bp.lost/bp.kept)

# write processed data to outfile
bp.to_csv(OUTF_bp)

# plot results
plot_mutations_bp(bp, OUTF_bp.replace('.csv',''))

# =============================================================================
# sections 
# =============================================================================

if RNA_type=='terminators':
    # get data
    sc = pd.read_csv(INF_sc_termNN, index_col=0)
    arnold_sc = calc_means(read_arnold(INF_sc_arnold), 'ARNold', ['section'])
    sc = pd.concat([sc, arnold_sc], ignore_index=True).reset_index(drop=True)
    
    # aligne values for bases    
    bases = sc[sc.mutations == 'base']
    bases = bases.rename(columns={'section':'base'}).drop(columns=['mutations'])
    sc = pd.merge(sc, bases, on=['model', 'k'])
    
    # calc differences
    sc['diff_abs'] = sc.base-sc.section
    sc['diff_rel'] = 1-(sc.section/sc.base)
    
    # write processed data to outfile
    sc.to_csv(OUTF_sc)
    
    #plot results
    plot_mutations_sc(sc, OUTF_sc.replace('.csv',''))
  
