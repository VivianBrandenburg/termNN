# -*- coding: utf-8 -*-

from tensorflow.keras.models import load_model
from termNN_tools import encoding , toolbox, validation, plots

import pandas as pd


ks = 10

res = pd.DataFrame()

for RNA_type in ['terminators', 'tRNAs']:
    
    # set right paths for RNA-type     
    PATH_out = f'results/{RNA_type}/validation/'
    PATH_val_data =  f'data/{RNA_type}/training/' 
    PATH_model =   f'models/{RNA_type}/'
    
    for m in toolbox.all_models:
        print(m['name'])
        valid_dic = {'acc':[], 'prec':[], 'rec':[], 'select':[], 'model':m['name'],
                     'model_k':[f'{m["name"]}_k{str(k)}' for k in range(ks)]}  
        
        # =====================================================================
        # calc validation for each model 
        # =====================================================================
        
        for k in range(ks):
            
            # get validation data
            get_data = encoding.choose_input(m['name'])
            y_pred, y_true = get_data(f'{PATH_val_data}k{str(k)}/val.csv')
            
            # get model
            model_INF = f'{PATH_model}{m["name"]}/{m["name"]}_k{str(k)}.h5'
            model = load_model(model_INF)
            
            # predict
            y_pred = model.predict(y_pred)
            
            # make validation
            y_pred = y_pred.reshape(len(y_pred),)
            y_pred[y_pred>0.5] = 1
            y_true, y_pred = y_true.astype(int), y_pred.astype(int)
            for key,value in validation.calc_validation(y_true, y_pred).items(): 
                valid_dic[key].append(value)
        # save results
        res = res.append(pd.DataFrame(valid_dic), ignore_index=True)
        
    
    # =========================================================================
    # if validating terminators: validate arnold data 
    # =========================================================================
    
    if RNA_type == 'terminators':
        arnold = validation.read_arnold('data/arnold/validation_data.txt')
        valid_dic = {'acc':[], 'prec':[], 'rec':[], 'select':[], 'model':'ARNold',
                     'model_k':[f'ARNold_k{str(k)}' for k in range(ks)]}  
  
        for k in range(ks):

            df = arnold[arnold.k == k]
            for key,value in validation.calc_validation(df.type, df.value).items(): 
                valid_dic[key].append(value)        
        res = res.append(pd.DataFrame(valid_dic), ignore_index=True)
        
    
    # =============================================================================
    # write out results
    # =============================================================================
    toolbox.check_and_make_dir(PATH_out)
    res.to_csv(PATH_out + 'validation.csv')
    
    plots.plot_performance(res, PATH_out+'validation')
    
