# -*- coding: utf-8 -*-

from termNN_tools import genomescan , plots
from termNN_tools.toolbox import check_and_make_dir
import sys


mode = sys.argv[1]

if mode not in ['encode_genome', 'scan_genome', 'find_kernels', 'calc_PPV']:
    raise ValueError("""you did not choose a valid running mode.
                     valid modes are: 'encode_genome', 'scan_genome', 'find_kernels', 'calc_PPV'""")



# dirs
DIR_genomic = 'data/terminators/ecoli_genome/'
DIR_models = 'models/terminators/'
DIR_res = 'results/terminators/genomescan/'
FILE_arnold = 'data/arnold/genomescan_ecoli_processed.csv'
DIR_PPV = DIR_res+'PPV/'

# variables
splits = 50
ks = 10
kernels_cutoff = 0.5


# read genomic data
genome = genomescan.get_genome(DIR_genomic + 'GCF_000005845.2_ASM584v2_genomic.fna')
gff= genomescan.get_gff(DIR_genomic + 'GCF_000005845.2_ASM584v2_genomic.gff')


for reverse, s in zip([False, True], ['fwd/', 'rev/']):
            
    # strand-specific files and dirs
    DIR_encode =  DIR_res+'encoded_genome/'+s
    DIR_res_s =  DIR_res+'results_single_nts/'+s
    DIR_kernels = DIR_res+'results_kernels/'+s
    check_and_make_dir(DIR_encode, DIR_res_s, DIR_kernels)
    

    # scan for terminators in genome  
    if mode == 'encode_genome': genomescan.encode_genome(genome, gff, splits, DIR_encode, reverse)
    if mode == 'scan_genome': genomescan.scan_genome(splits, ks, DIR_encode, DIR_models, DIR_res_s)
    
    # find kernels of model_output>cutoff
    if mode == 'find_kernels': genomescan.find_kernels(ks, kernels_cutoff, DIR_res_s, DIR_kernels, reverse)

    

# calc PPV of best kernels
if mode =='calc_PPV':
    res = genomescan.calc_PPV_best_kernels(DIR_res+'kernels/', FILE_arnold)
    res.to_csv(DIR_PPV+'PPV.csv')
    
    # plot results 
    plots.plot_genomescan(res, DIR_PPV+'PPV')

