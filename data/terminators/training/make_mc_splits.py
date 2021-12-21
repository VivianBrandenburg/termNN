# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 12:31:06 2021

@author: Vivian
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os


test = pd.read_csv('source_data/test_cleaned.csv', index_col=0)
train = pd.read_csv('source_data/train_cleaned.csv', index_col=0)

data = pd.concat([test, train])
pos = data[data.label == 1]
neg = data[data.label == 0]



def random_split(data):
    rest, train = train_test_split(data, test_size = 0.702)
    test, val = train_test_split(rest, test_size = 0.5)
    return train, test, val

def merge_set(pos, neg):
    data = pd.concat([pos, neg])
    data = data.sample(frac=1).reset_index(drop=True)
    return data


def make_split_data(pos, neg):
    pos_train, pos_test, pos_val = random_split(pos)
    neg_train, neg_test, neg_val = random_split(neg)
    
    train = merge_set(pos_train, neg_train)
    test  = merge_set(pos_test, neg_test)
    val   = merge_set(pos_val, neg_val)    
    return train, test, val
    
    

k = 10

for i in range(k):
    train, test, val = make_split_data(pos, neg)
    my_dir = 'k'+str(i)
    if not os.path.exists(my_dir):
        os.mkdir(my_dir)
    train.to_csv(my_dir+'/train.csv')
    test.to_csv(my_dir+'/test.csv')
    val.to_csv(my_dir+'/val.csv')
    
