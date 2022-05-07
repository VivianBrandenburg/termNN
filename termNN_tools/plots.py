# -*- coding: utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc

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




model_styles =[{'name':'CNN_onehot',  'col':'#8BB0CB', 'dash':[1,0], 'width':1.6, 'pre':False},
              {'name':'CNN_onehot_pretrained',  'col':'#274F66', 'dash':[5,2], 'width':1.3, 'pre':True},
              {'name':'CNN_matrix',   'col':'#FFDAA6', 'dash':[1,0], 'width':1.6, 'pre':False},
              {'name':'CNN_matrix_pretrained',   'col':'#eb9a28', 'dash':[5,2], 'width':1.3, 'pre':True}, 
              {'name':'LSTM_onehot',   'col':'#8BB68F', 'dash':[1,0], 'width':1.6, 'pre':False},
              {'name':'LSTM_onehot_pretrained',   'col':'#093315', 'dash':[5,2], 'width':1.3, 'pre':True}, 
              {'name':'ARNold',   'col':'dimgrey', 'dash': [1,1,1,5], 'width':1.3}, 
            ]




palette2 = {x['name']:x['col'] for x in model_styles}
dashes2 = {x['name']:x['dash'] for x in model_styles}



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
          'stretch_left': 'D',
          'stem_left': '^',
          'loop': 'X',
          'stem_right': 'v',
          'U_tail': 's',
          'strech_right': 's',
          'pad_right': 'H'}




palette_sections = {'pad_left': 'white',
          'A_tail': 'lightgrey',
          'stretch_left': 'lightgrey',
          'stem_left': '#E76A3C',
          'loop': 'dimgrey',
          'stem_right': '#E76A3C',
          'U_tail': 'lightgrey',
          'strech_right': 'lightgrey',
          'pad_right': 'white'}








plotcolor = 'dimgrey'


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
            


def fix_axis(ax):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
        ax.spines[axis].set_color(plotcolor)
    ax.tick_params(width=0.5, color=plotcolor)
    


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
 
    
    



def plot_sendseq(aucs, prerec, plottype, DIR_plot):
        
       
        
    # prep plot data
    if plottype == 'LSTM':
        plot_aucs = aucs[aucs.model.str.contains('LSTM')]
        plot_prerec = prerec[prerec.model.str.contains('LSTM')]
        # plot_avscore = pd.read_csv('ju2019_aroundSEndSeqHits.csv')
        model_selected = ['LSTM_onehot', 'LSTM_onehot_pretrained']
        plotname = 'ju2019_LSTM'
        
    if plottype == 'CNN':
        plot_aucs = aucs[~aucs.model.str.contains('LSTM')]
        plot_prerec = prerec[~prerec.model.str.contains('LSTM')]
        # plot_avscore = pd.read_csv('ju2019_aroundSEndSeqHits.csv')
        model_selected = ['CNN_onehot', 'CNN_onehot_pretrained', 'CNN_matrix', 'CNN_matrix_pretrained']
        plotname = 'ju2019_CNN'
    

   
    
    # prep plot style
    labelsize=8
    linewidth = 1
    plotcolor = 'dimgrey'
    sns.set_style("ticks")
    rc('xtick', labelsize=labelsize, color=plotcolor) 
    rc('ytick', labelsize=labelsize, color=plotcolor )
    rc('legend', fontsize=labelsize,  labelcolor=plotcolor)
    
    dashes2 = {m['name']:m['dash'] for m in model_styles }
    plot_labels=[model_labels[x] for x in model_selected]
    
    
    
    
    
    # make plot layout
    fig, axs = plt.subplots(2, 2, figsize=(7.5,5.7), gridspec_kw={'wspace':0.27, 'hspace':0.46})
    (ax1, ax2), (ax3, ax4) = axs
    
    
    
    # ax1: average scores around tts 
    # was removed because data size was too large
    
    
    
    # ax2: area under precision-recall-curve for different thresholds
    sns.lineplot(ax=ax2, data=plot_aucs, x='max_dist', y='auc', hue='model',
                  markers=['o', 's']*int(len(model_selected)/2), style='model', markersize=5, 
                  palette = palette2, dashes=dashes2, linewidth=linewidth)
    ax2.set_xlabel('distance treshold', size=labelsize, color=plotcolor)
    ax2.set_ylabel('AUC', size=labelsize, color=plotcolor)
    
    legend_labels, _= ax2.get_legend_handles_labels()
    ax2.legend(legend_labels, plot_labels, frameon=False)
        
    ax2.vlines(x=35, ymin=0.01, ymax=0.6, linestyle='--', colors='grey', linewidth=linewidth)
    
    
    # ax3: precision-recall-curve for distance = 35 nt 
    prerec_dist = plot_prerec[plot_prerec.max_dist == 35]
    for model in [x for x in model_styles if x['name'] in model_selected]:
        for k in range(10):
    
                model_k = model['name']+'_'+str(k)
                res_ms = prerec_dist[prerec_dist.model_k == model_k]
                
                if k < 9:
                    ax3.plot(res_ms.rec, res_ms.prec, 
                            dashes = model['dash'], linewidth=model['width']*0.5, 
                            color=model['col'])
                else:
                    ax3.plot(res_ms.rec, res_ms.prec, 
                              dashes = model['dash'], linewidth=model['width']*0.5,
                              color=model['col'],
                              label=model_labels[model['name']])  
                    
                    
    res_arnold = prerec[(prerec.model=='arnold') & (prerec.max_dist == 35)]
    ax3.plot(res_arnold.rec, res_arnold.prec,   
             'o', markersize=5, linewidth=model['width'],
             color=palette2['ARNold'],  label='ARNold', alpha=0.7) 
    ax3.set_title('max dist = ' + str(35)  + ' nt', size=labelsize, color=plotcolor)
    ax3.legend(prop={"size":labelsize}, 
               bbox_to_anchor=(0.3,-0.6), loc=6, borderaxespad=0., 
               labelcolor=plotcolor, frameon=False)                        
    ax3.set_xlabel('recall', size=labelsize, color=plotcolor)
    ax3.set_ylabel('precision', size=labelsize, color=plotcolor)
        
    
    
    # ax4: area under precision-recall curve at dist = 35 nt
    sns.swarmplot(ax=ax4, data=plot_aucs[plot_aucs.max_dist == 35], x='model', y='auc', 
                  dodge=True,  size=6, palette = palette2, 
                  zorder=1, alpha=0.7)
    sns.pointplot(ax=ax4, data=plot_aucs[plot_aucs.max_dist == 35], x='model', y='auc',
                  palette=['dimgrey'], zorder=0, capsize=0.15, scale=0.4, 
                  markers='s', errwidth=0.7)
    ax4.set_xticklabels([model_labels[x.get_text()].replace(' ', '\n') for x in  ax4.get_xticklabels()])
    
    ax4.set_xlabel('')
    ax4.set_ylabel('AUC', size=labelsize, color=plotcolor)
    ax4.set_title('max dist = ' + str(35)  + ' nt', size=labelsize, color=plotcolor)
    # ax4.set_ylim(0.35, 0.61)
    
    
    
    # fix axes styles 
    for ax in [ax1, ax2, ax3, ax4]:
        fix_axis(ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
    
    # save and plot
    plt.savefig(DIR_plot + plotname  + '.svg',  bbox_inches = 'tight')
    plt.savefig(DIR_plot + plotname + '.png',  bbox_inches = 'tight', dpi = 600)
    plt.show()
    
