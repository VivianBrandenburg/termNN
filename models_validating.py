# -*- coding: utf-8 -*-

from tensorflow.keras.models import load_model
from termNN_tools import encoding, toolbox, plots
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score
import pandas as pd



ks = 10
arnold= toolbox.read_arnold_validation('data/arnold/validation_data.txt')


RNA_type = ['terminators']

# set right paths for RNA-type     
PATH_out = f'results/{RNA_type}/validation/'
PATH_val_data =  f'data/{RNA_type}/training/' 
PATH_model =   f'models/{RNA_type}/'


# calc precicion-recall-curve for each model 
prerec_table = pd.DataFrame()
metrics = pd.DataFrame()

for m in toolbox.all_models:

    for k in range(ks):
        print(m['name'], k)
        
        # get validation data
        get_data = encoding.choose_input(m['name'])
        y_pred, y_true = get_data(f'{PATH_val_data}k{str(k)}/val.csv')

        # get model prediction 
        model_INF = f'{PATH_model}{m["name"]}/{m["name"]}_k{str(k)}.h5'
        model = load_model(model_INF)
        y_pred =list( model.predict(y_pred).reshape(len(y_pred),))
        y_true = list(y_true)
        
        # calc precision-recall-curves
        prec, rec, thresh = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        prerec_table = prerec_table.append(
            pd.DataFrame({'model': m['name'], 'k':k, 'prec':prec, 'rec':rec})
            , ignore_index=True)
        
        # f1-score, precision,recall and area under curve for precision-recall-curve
        y_pred = [1 if x>=0.5 else 0 for x in y_pred]
        
        res = pd.DataFrame({'model': m['name'],  'k':[k],
                            'auc': auc(rec, prec),
                            'f1': f1_score(y_true, y_pred),
                            'precision': precision_score(y_true, y_pred),
                            'recall': recall_score(y_true, y_pred)
                            })

        metrics = metrics.append(res, ignore_index=True)

        
        
# f1-score , precision, recall for arnold
for k in range(ks):
    arnold_k = arnold[arnold.k == k]
    y_true, y_pred = list(arnold_k.y_true), list(arnold_k.y_pred)
    res = pd.DataFrame({'model': 'ARNold',  'k':[k],
                        'auc': 'nan',
                        'f1': f1_score(y_true, y_pred),
                        'precision': precision_score(y_true, y_pred),
                        'recall': recall_score(y_true, y_pred)
                        })
    metrics = metrics.append(res, ignore_index=True)
    

# write results 
prerec_table.to_csv(PATH_out + 'precision_recall_curve.csv')
metrics.to_csv(PATH_out + 'metrics.csv')


# plot results
plots.plot_validation(prerec_table, metrics, 'CNN', PATH_out + 'validation_plots', RNA_type)
plots.plot_validation(prerec_table, metrics, 'LSTM', PATH_out + 'validation_plots_LSTMs', RNA_type)
