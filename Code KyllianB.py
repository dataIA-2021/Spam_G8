# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:53:59 2022

@author: KyllianB
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
#%% Test types de séries

df_test = pd.DataFrame({'Lettre':['A', 'B', 'C', 'D', 'E'], 
                        'Voyelle':[True, False, False, False, True],
                        'Itérations':[3, 1, 0, 2, 5], 
                        'Ratio':[3/15 , 1/15 , 0/15 , 2/15 , 5/15]})

#%%Test pour la fonction

df = pd.read_csv('spam_clean.csv')
Target = 'label'
X = df.drop(columns = Target)

print(X)

#%%

# data
import pandas as pd
# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
# Pipeline and model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
# Score of models
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



def classification(df, Target):
    '''
    Parameters
    ----------
    df : Pandas.DataFrame
        DataFrame sur lequel on souhaite entraîner et appliquer le modèle
    Target : str
        Nom de la colonne contenant la  classification
        
    Random_seed : int, compris entre 0 et 999
        Seed du random_state pour le train_test_split. The default is 42.
    Sortie :
        Affiche le score du modèle et la matrice de confusion
    '''
    model = GradientBoostingClassifier()
    
    Target_Encoder = LabelEncoder()
    y = Target_Encoder.fit_transform(df[Target])
    
    X = df.drop(columns = Target)
    
    column_cat = (df.select_dtypes(include = ['object']).drop(columns = Target)).columns
    column_num = (df.select_dtypes(include = ['int', 'float'])).columns
    
    transformer_cat = Pipeline(steps = [
        ('imputation', SimpleImputer(strategy='most_frequent')),
        ('label encoder', LabelEncoder()), 
        ('scaler', MinMaxScaler())])
    
    transformer_num = Pipeline(steps = [
        ('imputation', SimpleImputer(strategy='median')),
        ('scaling', MinMaxScaler())])
    
    preparation = ColumnTransformer(transformers =
                                    [('data_num', transformer_num, column_num),
                                     ('data_cat', transformer_cat, column_cat)])
    
    pipe_model = Pipeline(steps=
                          [('preparation', preparation),
                           ('model', model)])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y)
    
    pipe_model.fit(X_train, y_train)
    y_pred = pipe_model.predict(X_test)
    
    score = accuracy_score(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
    
    return(score)


df = pd.read_csv('spam_features.csv')
print(round(classification(df.drop(columns = 'content'), 'label'), 4))


#%%

#scores = []
#for k in range (1000):
#    scores.append(classification(df.drop(columns = 'content'), 'label'))
print(pd.Series(scores).describe())

#%%

#df.to_csv('spam_features.csv',index = False)