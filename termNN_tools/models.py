# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential, metrics
from tensorflow.keras.layers import Dense, Flatten, Dropout



# =============================================================================
# variables
# =============================================================================
METRICS = [
            metrics.BinaryAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')]


optim = 'adamax'

    
    
# =============================================================================
# build models
# =============================================================================

def CNN_matrix(seq_len):
    from tensorflow.keras.layers import MaxPooling2D, Conv2D
    set_seeds()
    model = Sequential(
        [Conv2D( filters=30, kernel_size=10, activation='relu', input_shape=(seq_len,seq_len,1)), 
        MaxPooling2D(pool_size=5),
        Dropout(0.2),
        Flatten(),
        Dense(360, activation='relu'),
        Dense(30, activation='sigmoid'),
        Dense(1,activation="sigmoid")])
    model.compile(loss='binary_crossentropy',optimizer=optim,
                  metrics=METRICS)
    return model
    


def CNN_onehot(seq_len):
    from tensorflow.keras.layers import MaxPooling1D, Conv1D
    set_seeds()
    model = Sequential(
        [Conv1D( filters=30, kernel_size=10, activation='relu', input_shape=(seq_len,4)),
         MaxPooling1D(pool_size=5),
         Dropout(0.2),
         Flatten(),
         Dense(360, activation='relu'),
         Dense(30, activation='sigmoid'),
         Dense(1,activation="sigmoid")])
    model.compile(loss='binary_crossentropy',optimizer=optim,
                  metrics=METRICS)
    return model



def LSTM_onehot(seq_len):
    from tensorflow.keras.layers import LSTM, Bidirectional
    set_seeds()
    model = Sequential(
        [Bidirectional(LSTM(10, input_shape=(seq_len,4),  return_sequences=True)),
        Dropout(0.3),
        Flatten(),
        Dense(300, activation='relu'),
        Dense(30, activation='relu'),
        Dense(1,activation="sigmoid")])
    model.compile(loss='binary_crossentropy',optimizer=optim,
              metrics=METRICS)
    return model



# =============================================================================
# early stopping 
# =============================================================================

def early_stop(patience):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', verbose=1, patience=patience,
        mode='max', restore_best_weights=True)
    return early_stopping



def get_variables(model_name, RNA_type, pretraining=False):
    model = model_name.replace('_pretrained', '')
    
    if model=='CNN_onehot':    stop, batch = early_stop(20), 40
    elif model=='CNN_matrix':  stop, batch = early_stop(5) , 40
    elif model=='LSTM_onehot': stop, batch = early_stop(10), 60

    if RNA_type == 'tRNAs':
        if not pretraining: stop = early_stop(10)
        else:
            print('loading pretraining variables for tRNAs')
            if model=='CNN_onehot':    stop, batch = early_stop(20), 30
            elif model=='CNN_matrix':  stop, batch = early_stop(10), 50
            elif model=='LSTM_onehot': stop, batch = early_stop(20), 50
    return stop, batch
        


    
def set_seeds():
    np.random.seed(202)
    tf.random.set_seed(202)
    
    
# =============================================================================
# choose model and encoding function
# =============================================================================

def choose_model(model_name):
    model = model_name.replace('_pretrained', '')
    if model=='CNN_onehot':    make_model = CNN_onehot
    elif model=='CNN_matrix':  make_model = CNN_matrix
    elif model=='LSTM_onehot': make_model = LSTM_onehot
    else:  raise ValueError('there is no model for your chosen input type here')
    return make_model
        

