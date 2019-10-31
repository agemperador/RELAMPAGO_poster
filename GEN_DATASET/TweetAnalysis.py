#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 01:06:35 2019

@author: mech
"""

################################################################
####														####
####		Functions to normalize raw text (our data)		####
####														####
################################################################

import numpy as np
import pandas as pd

#%%  Hermes: God of Language

def hermes(DATAFRAME, name_MSJ, name_USER, number_spam):
    """
        Use of Hermes in raw text:
        1) Transfoms raw text to ASCII, translates emojis and allows to use them as plain text
        2) Deletes unimportant words
        3) Deletes spam tweets
        4) Deletes symbols
    
        DATAFRAME: dataframe to analyze
        name_MSJ: name of the column from the DATAFRAME with the raw tweets, enter as string
        name_USER: name of the column from the DATAFRAME with the user id, enter as string
        number_spam: threshold to consider a Twitter profile as spam
        
        """
        
    #----- SPAM REMOVAL -----#
    
    cant_users = list(set(DATAFRAME[name_USER]))
    
    spam = []
    for i in cant_users:
        h = list(DATAFRAME[name_USER]).count(i)
        if h >= number_spam: spam.append(i)
    
    aux = []       
    for i in spam:
        aux = np.append(aux, np.where(DATAFRAME[name_USER] == i)[0])
    
    DATAFRAME = DATAFRAME.drop(aux, axis = 0)
    
    #----- NORMALIZATION -----#
    
    mensaje = list(DATAFRAME[name_MSJ])
    lenght = len(mensaje)
    
    lista_ascii = [['\\xe1', '\\xe9', '\\xed', '\\xf3', '\\xfa', '\\xf1', '\\xd1', '\\xc1','\\xc9', '\\xcd', '\\xd3', 
                 '\\xda', '\\xa1', '\\xbf', '\\xdc', '\\xfc', '\\xb4', '\\xba', '\\xaa', '\\xb7','\\xc7','\\xe7',
                 '\\xa8', '\\xb0C', '\\n', '\\xb0c', '\\xbb', 'xb0', '\\ufe0f'],['a', 'e', 'i', 'o', 'u', 'ñ', 'Ñ', 'A', 'E', 'I', 'O', 'U', '', '', 'Ü', 'ü', '', 
                           ' ', '', '','Ç','ç','', ' ', ' ', ' ', ' ', ' ', ' ']]
    lista_simb = ['!','\'', '\"', "|", '$', "%", "&", "(", ")", "=", "?", "+",'/', ";", "_", "-", "1","2","3","4","5","6","7","8","9","0", '*',',', '.', ':', '#']
    
    ign_palabras = ['mbar','cdu','s','la','lo','st','que','este', 'me','t','p','el','en','h','temp','hpa','km','mm',"su","vos",'que',"re","xq","le","te","tu","soy","sos","mi","da","o","x","les","me","d","q", "lo", "los", "mi", "son", "a", "el", 
                    "un","la", "una","en","por","para", 'las',"ante", "al", 'me',"rt", "del", "y", "se", "de", "que", "sus", "ha", "es", "esta"]
    
    for i in range(lenght):
        
        ## Convierto mayúsculas en minúsculas
        mensaje[i] = mensaje[i].lower()
        
        ## Saco las menciones y otras cosas
        txt = mensaje[i].split()
        
        for j in range(len(txt)):
            if ('@' in txt[j]) and ('RELAMPAGO2018' not in txt[j]) and ('RELAMPAGO_edu' not in txt[j]) or ('jaj' in txt[j]) or ('https' in txt[j]): txt[j]=''
        
        
        mensaje[i] = " ".join(txt)
        
        ## Reemplazo símbolos
        for h in range(len(lista_simb)):
            mensaje[i] = mensaje[i].replace(lista_simb[h], ' ')
            
        ## Convierto el msj a ASCII, reemplazo los errores de decodificación y agrego un espacio antes de cada decodificación
        mensaje[i] = mensaje[i].encode('unicode-escape').decode('ASCII')+" "     
        
        for j in range(len(lista_ascii[0])):
            mensaje[i] = mensaje[i].replace(lista_ascii[0][j], lista_ascii[1][j])
        
        mensaje[i] = mensaje[i].replace('\\', ' \\')
        
        for j in range(len(ign_palabras)):
            mensaje[i] = mensaje[i].replace(" "+ign_palabras[j]+" ", ' ')
        
    DATAFRAME['msj_norm'] = mensaje
    return(DATAFRAME)
