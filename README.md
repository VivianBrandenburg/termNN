# termNN
models for detection of intrinsic transcription terminators and tRNAs in bacteria

## package requirements

tensorflow v2.1  
h5py       v2.10.0  
numpy      v1.19.5  


## train and validate models for terminators + tRNAs
**models_training.py**: Pretrain and train CNN and LSTM models.  
**models_validating.py**: Validate all trained models, write output to results/validation, plot results.  

## analyse the impact of structure of terminators + tRNAs, and the impact of strcuture of terminators
**mutate_basepairs.py**: Mutate an increasing number of base pairs in tRNAs and terminators, feed them into all trained models, write output to results.   
**mutate_sections.py**: Mutate half of the nucleotides of each section of terminators, feed them into all trained models, write output to results.  
**mutate_processing.py**: Summarize results of the mutation experiments, write output to results/mutation, plot results.  

## scan genome for terminators
**scan_genome.py**: Use with one or more options of ['encode_genome', 'run_scan', 'find_kernels', 'calc_PPV']:

- option 'encode_genome': Splits the input genome into overlapping subsequences of 75 base pairs and encodes them for the scan. The resulting encoded sequences are written to directory genomescan/encoded_genome.  
- option 'run_scan': Runs the actual scan. Reads encoded sequences from genomescan/encoded_genome and writes all model output of all models to genomescan/results_single_nts.  
- option 'find_kernels': searches for stretches of nucleotides with a model output greater 0.5 and combines them to one kernel. Reads model output from genomescan/results_single_nts, writes results to genomescan/results_kernels.  
- option 'calc_PPV': Calculates PPV for genome scan. 
