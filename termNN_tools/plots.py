# -*- coding: utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# styles
# =============================================================================


model_labels = {'CNN_onehot': 'one-hot CNN',
          'CNN_onehot_pretrained': 'one-hot CNN pre-trained',
          'CNN_matrix': 'matrix CNN', 
          'CNN_matrix_pretrained': 'matrix CNN pre-trained',
          'LSTM_onehot': 'one-hot LSTM', 
          'LSTM_onehot_pretrained': 'one-hot LSTM pre-trained',
          'ARNold':'ARNold'}


val_labels = {'acc': 'accuracy',  'prec': 'precision',
                         'rec': 'sensitivity', 'select': 'specificity'}


palette_blues = {'CNN_onehot': '#084068', 'CNN_onehot_pretrained': '#084068', 
              'CNN_matrix': '#5E799D', 'CNN_matrix_pretrained': '#5E799D',
                'LSTM_onehot': '#AAB5CC',  'LSTM_onehot_pretrained': '#AAB5CC', 
                'ARNold': '#B9CC95', 'ARNold_paper': '#7B8569' }




order = ['CNN_onehot', 'CNN_onehot_pretrained', 
          'CNN_matrix', 'CNN_matrix_pretrained', 
          'LSTM_onehot', 'LSTM_onehot_pretrained',
          'ARNold']


markers_sections = {'pad_left': 'o',
          'A_tail': 'D',
          'stem_left': '^',
          'loop': 'X',
          'stem_right': 'v',
          'U_tail': 's',
          'pad_right': 'H'}



palette_sections = {'pad_left': '#E2C987',
          'A_tail': '#315361',
          'stem_left': '#E76A3C',
          'loop': '#9C3C19',
          'stem_right': '#E76A3C',
          'U_tail': '#315361',
          'pad_right': '#E2C987'}





# =============================================================================
# general functions
# =============================================================================

def make_hatches(myplot, hatches):
    for i,thisbar in enumerate(myplot.patches):
        thisbar.set_hatch(hatches[i], )
        
def save_figs(PLOTDIR):
    plt.savefig(PLOTDIR + '.svg', bbox_inches = 'tight')
    plt.savefig(PLOTDIR + '.png', bbox_inches = 'tight',  dpi=300)
    


def sns_styles(change_style={}):
    style = {'style':'ticks', 'context':'notebook',
             'font_scale':1.3, 'linewidth':0.8,
             'linewidth': 0.8,
             'figsize':(7,5)}
    
    for key, value in change_style.items():
        style[key] = value
    
    sns.set(rc={'figure.figsize':style['figsize']})
    sns.set_style(style['style'])    
    sns.set_context(style['context'], font_scale=style['font_scale'], 
                    rc={"lines.linewidth": style['linewidth']})
            
        
# =============================================================================
# specific functions     
# =============================================================================



def plot_performance(data, outname):
    if outname.find('terminator') != -1:
        arnold_2 = {'rec': 0.878,  'select': 0.953, 'model': "ARNold_paper"}
        data = data.append(arnold_2, ignore_index=True)
    
    data = pd.melt(data, id_vars=['model', 'model_k'])
    sns_styles({'figsize':(5.8,1.5), 'font_scale':0.7, 'linewidth':0.3})
    myplot = sns.barplot(data=data,  x='variable', y='value', hue='model', palette=palette_blues,
                         order=['acc', 'rec', 'select'], capsize=0.03)
    myplot.set_xticklabels([val_labels[x.get_text()] for x in  myplot.get_xticklabels()])
    make_hatches(myplot,['', '', '',  '\\\\','\\\\', '\\\\',  ]*3+ ['X']*3+ ['X']*4)
    plt.ylim(0.5,)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel(''); plt.ylabel('')
    save_figs(outname)
    plt.show()
    




def plot_genomescan(data, outname):
    sns_styles({'figsize':(10,5), 'style':'whitegrid'})
    scan = sns.barplot(data=data,  x='model', y='PPV', 
                        order=order, capsize=.1, palette=palette_blues)
    scan.set_xticklabels([model_labels[x.get_text()].replace(' ', '\n') for x in scan.get_xticklabels()])
    make_hatches(scan,['', '\\\\']*3+['X'])
    plt.xlabel('')
    plt.ylim(0, 0.8)
    save_figs(outname)
    plt.show()
    



def plot_mutations_bp(data, outname):
    sns_styles({'context':'paper', 'figsize':(3.7,2.5), 'font_scale':1, 'style':'whitegrid', 'linewidth':2})
    data['pretrained']=data.model.str.contains('pretrained')
    
    myplot=sns.lineplot(data=data, hue='model', x='mutations', y='diff_rel', palette=palette_blues, style='pretrained', legend='full')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # place legend on side
    plt.xlabel('mutated base pairs')
    plt.ylabel('impact of basepairing')
    save_figs(outname)
    plt.show()
    


def plot_mut_sc_average(data):
    mymeans = pd.DataFrame()
    model_nums = data.model.unique()
    model_nums = {key:value for key, value in zip(model_nums, range(len(model_nums)))}

    for model, value in model_nums.items():
        for section in data.mutations.unique():
            select =  data[(data.model==model)& (data.mutations==section)]
            res = {'model':model, 'section': section,
                    'model_num': value,
                    'diff_rel': select.diff_rel.mean()}
            mymeans = mymeans.append(res, ignore_index=True)
    return mymeans
    


def plot_mutations_sc(data, outname):
    data = data[data.mutations != 'base'].reset_index(drop=True)
    data.loc[data.mutations=='stretch_left', 'mutations'] = 'A_tail'
    data.loc[data.mutations=='strech_right', 'mutations'] = 'U_tail'
    data_means = plot_mut_sc_average(data)
    
    sns_styles({'style': 'whitegrid', 'context':'notebook', 'figsize':(5,6), 'font_scale':1})
    myplot = sns.scatterplot(data=data_means, x='model', y='diff_rel',hue='section', style='section', 
                s=150, markers=markers_sections, palette=palette_sections, alpha=0.85, linewidth=0.6, edgecolor='black')
    myplot.set_xticklabels([x.replace(' ', '\n') for x in model_labels.values()])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
    save_figs(outname)
    plt.show()
 