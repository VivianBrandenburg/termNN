# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc


def calc_content(data):
    content = {key:[] for key in range(len(data[0]))}
    for seq in data:
        for n, nt in enumerate(seq):
            content[n].append(nt)
    content = {key:[1 if x in ["G", "C"] else 0 for x in value] for key, value in content.items()}
    content = {key:sum(value)/len(value)*100 for key, value in content.items()}
    return content




def content_posneg(data):
    pos = list(data[data.label == 1].sequence)
    neg = list(data[data.label == 0].sequence)
    c_pos = calc_content(pos)
    c_neg = calc_content(neg)
    c_data = pd.DataFrame({'position':list(c_pos.keys()) + list(c_neg.keys()), 
                           'GC-content [%]':  list(c_neg.values()) + list(c_pos.values()), 
                           'sequence': ['neg']*75 + ['term']*75})
    return c_data

    
# =============================================================================
# read in data
# =============================================================================

data_path = '../../../data/terminators/'

pretrain = pd.concat([pd.read_csv(data_path + 'pretraining/test.csv', index_col=0),
                      pd.read_csv(data_path + 'pretraining/train.csv', index_col=0),
                      ])

maintrain = pd.concat([pd.read_csv(data_path + 'training/k0/val.csv', index_col=0), 
                       pd.read_csv(data_path + 'training/k0/train.csv', index_col=0), 
                       pd.read_csv(data_path + 'training/k0/test.csv', index_col=0),
                       ])


# =============================================================================
# get gc content 
# =============================================================================
c_pretrain = content_posneg(pretrain)
c_pretrain['label'] = ['artificial terminator' if x == 'term' else 'artificial negative' for x  in c_pretrain['sequence']]

c_maintrain  = content_posneg(maintrain)
c_maintrain['label'] = ['terminator' if x == 'term' else 'negative' for x  in c_maintrain['sequence']]

data = pd.concat([c_maintrain, c_pretrain], ignore_index=True)


# =============================================================================
# plotting
# =============================================================================


def fix_axis(ax):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
        ax.spines[axis].set_color('dimgrey')
    ax.tick_params(width=0.5, color='dimgrey')
    
plotcolor = 'dimgrey'



dashes = {'negative': [1,0] , 'terminator': [1,0], 
              'artificial negative': [4,1], 'artificial terminator': [4,1] }



colors = {'negative': 'lightgrey' , 'terminator':'grey', 
              'artificial negative': 'lightgrey', 'artificial terminator': '#E76A3C' }




sns.set_style("ticks")
rc('xtick', labelsize=8, color=plotcolor) 
rc('ytick', labelsize=8, color=plotcolor) 
rc('legend',  labelcolor=plotcolor)
  

fig,ax=plt.subplots(figsize=(7.5,4))
g = sns.lineplot(ax=ax, data=data, x='position', y='GC-content [%]', 
             hue='label', palette=colors,  
             style='label', dashes=dashes, 
             hue_order=['artificial negative', 'artificial terminator', 'negative', 'terminator'],
             linewidth=2, alpha=0.9)
plt.legend(title=None, bbox_to_anchor=(1.01, 0.77), loc=0, borderaxespad=0., frameon=False )
ax.set_xlabel('position', color=plotcolor)
ax.set_ylabel('GC-content [%]', color=plotcolor)
sns.despine()
fix_axis(ax)
ax.tick_params( color=plotcolor)


plt.savefig('GCcontent.svg',  bbox_inches='tight')
plt.savefig('GCcontent.png', dpi=600,  bbox_inches='tight')
plt.show()
