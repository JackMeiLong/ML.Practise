# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 21:24:56 2016

@author: mellon
"""

import pandas as pd
import numpy as np

class Boosting(object):
    
    def __init__(self):
        self.train=[]
        self.label=[]
        self.test=[]
        self.weight_sample=[]
        self.weight_classifier=[]
        self.features=[]
    
    def importdata(self,path):
        df=pd.read_csv(path)
        self.train=df.values[:,:-1]
        self.label=df.values[:,-1]
        self.features=df.columns
        
    def Boosting(self,estimator,k):
        m,n=np.shape(self.train)
        weight_sample=np.ones((m,1))
        weight_classifier=np.zeros((k,1))
        
        
        
        