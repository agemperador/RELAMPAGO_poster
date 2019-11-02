#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:58:36 2019

@author: mech
"""

############################
####                    ####
####    Bag Of Words    ####
####                    ####
############################

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

#%% ----- BOW Class ----- %%#

class BoWClassifier(nn.Module): # inheriting from nn.Module!
    
    def __init__(self, num_labels, vocab_size):
        
        super(BoWClassifier, self).__init__()

        self.linear = nn.Linear(vocab_size, num_labels)
        
        
    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(bow_vec), dim=1)
    
    def make_bow_vector(sentence, word_to_ix):
        vec = torch.zeros(len(word_to_ix))
        txt = sentence.split()
        for word in txt:
            vec[word_to_ix[word]] += 1
        return vec.view(1, -1)

    def make_target(label, label_to_ix):
        
        return torch.LongTensor([label_to_ix[label]])

def bow(DATAFRAME, columna):
    word_to_ix = {}

    # El diccionario tiene que tener todas las palabras posibles
    datos = DATAFRAME[columna]

    ## Cada palabra que encuentra la guarda en un diccionario
    for sent in datos:
        txt = sent.split()
        for word in txt:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    VOCAB_SIZE = len(word_to_ix)
    NUM_LABELS = 2
    
    model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)
    vec = torch.zeros(len(word_to_ix))
    for i in datos:
        bow_vector = BoWClassifier.make_bow_vector(i, word_to_ix)
        vec = vec.add(bow_vector)

    vec=vec[0].tolist()

    ## Genero un dataframe que me pemite ordenar de mayor a menor respecto de la columna Frecuencia
    output = pd.DataFrame()
    output['Diccionario'] = list(word_to_ix.keys())
    output['Frecuencia'] = vec
    output['Num'] = word_to_ix.values()
    output = output.sort_values(by=['Frecuencia'], ascending=False)
    return output
