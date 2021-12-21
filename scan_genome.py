# -*- coding: utf-8 -*-

from termNN_tools import genomescan , plots
from termNN_tools.toolbox import check_and_make_dir
import sys



# check for right running mode
valid_modes = ['encode_genome', 'run_scan', 'find_kernels', 'calc_PPV']
if not set(sys.argv) & set(valid_modes):
    raise ValueError("\n\n\nyou did not choose a valid running mode.\nvalid modes are: 'encode_genome', 'run_scan', 'find_kernels', 'calc_PPV'\n\n")
    




# input directories
DIR_genomic = 'data/terminators/ecoli_genome/'
DIR_models = 'models/terminators/'
FILE_arnold = 'data/arnold/genomescan_ecoli_processed.csv'


#output directories
DIR_res = 'results/terminators/genomescan/'
DIR_encode =  DIR_res+'encoded_genome/'
DIR_res_single =  DIR_res+'results_single_nts/'
DIR_kernels = DIR_res+'results_kernels/'
DIR_PPV = DIR_res+'PPV/'


# variables
splits = 50
ks = 10
kernels_cutoff = 0.5


# read genomic data
genome = genomescan.get_genome(DIR_genomic + 'GCF_000005845.2_ASM584v2_genomic.fna')
gff= genomescan.get_gff(DIR_genomic + 'GCF_000005845.2_ASM584v2_genomic.gff')


for reverse, dirname in zip([False, True], ['fwd/', 'rev/']):
            
    # strand-specific files and dirs
    DIR_encode_s =  DIR_encode + dirname
    DIR_res_single_s =  DIR_res_single + dirname
    DIR_kernels_S = DIR_kernels + dirname
    
    # encode and scan genom
    if 'encode_genome' in sys.argv: genomescan.encode_genome(genome, gff, splits, DIR_encode_s, reverse)
    if 'run_scan' in sys.argv: genomescan.scan_genome(ks, DIR_encode_s, DIR_models, DIR_res_single_s)
    # find kernels  
    if 'find_kernels' in sys.argv: genomescan.find_kernels(ks, kernels_cutoff, DIR_res_single_s, DIR_kernels_S, reverse)


# calc PPV of best kernels
if 'calc_PPV' in sys.argv:
    res = genomescan.calc_PPV_best_kernels(DIR_kernels, FILE_arnold)
    check_and_make_dir(DIR_PPV)
    res.to_csv(DIR_PPV+'PPV.csv')
    
    # plot results 
    plots.plot_genomescan(res, DIR_PPV+'PPV')

