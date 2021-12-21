# -*- coding: utf-8 -*-

from termNN_tools import genomescan , plots
from termNN_tools.toolbox import check_and_make_dir


# dirs
DIR_genomic = 'data/terminators/ecoli_genome/'
DIR_models = 'models/terminators/'
DIR_res = 'results/terminators/genomescan/'
FILE_arnold = 'data/arnold/genomescan_ecoli_processed.csv'

# variables
splits = 50
ks = 10
kernels_cutoff = 0.5


# read genomic data
genome = genomescan.get_genome(DIR_genomic + 'GCF_000005845.2_ASM584v2_genomic.fna')
gff= genomescan.get_gff(DIR_genomic + 'GCF_000005845.2_ASM584v2_genomic.gff')


for reverse, s in zip([False, True], ['fwd/', 'rev/']):
            
    # strand-specific files and dirs
    DIR_encode =  DIR_res+'encoded/'+s
    DIR_res_s =  DIR_res+'all_results/'+s
    DIR_kernels = DIR_res+'kernels/'+s
    check_and_make_dir(DIR_encode, DIR_res_s, DIR_kernels)
    


# #### the following steps are time-consuming!
# #### un-comment them if you want to start the genome-scan from scratch 
# #### >>> 

#     # scan for terminators in genome  
#     genomescan.encode_genome(genome, gff, splits, DIR_encode, reverse)
#     genomescan.scan_genome(splits, ks, DIR_encode, DIR_models, DIR_res_s)
    
#     # find kernels of model_output>cutoff
#     genomescan.find_kernels(ks, kernels_cutoff, DIR_res_s, DIR_kernels, reverse)

# #### <<<    
    

# calc PPV of best kernels
DIR_PPV = DIR_res+'PPV/'; check_and_make_dir(DIR_PPV)
res = genomescan.calc_PPV_best_kernels(DIR_res+'kernels/', FILE_arnold)
res.to_csv(DIR_PPV+'PPV.csv')


# plot results 
plots.plot_genomescan(res, DIR_PPV+'PPV')
