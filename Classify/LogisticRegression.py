# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:15:34 2016

@author: meil

Logistic Regression

http://blog.csdn.net/Fishmemory/article/details/51603836

"""

import numpy as np
import pandas as pd
import math

class LogisticRegression(object):
    
    def __init__(self):
        self.train=[]
        self.train_label=[]
        self.train_df=[]
        self.weight=[]
        self.ratio=0.001
        self.maxcycle=500
        self.test=[]
        self.test_label=[]
        self.features=[]
    
    def loaddataset(self,path):
        df=pd.read_csv(path)
        m=np.shape(df.values)[0]
        self.train=df.values[:,:-1]
        self.train_label=df.values[:,-1]
        self.train_df=df[:int(0.7*m)]
        self.test=df.values[0.7*m:,:-1]
        self.test_label=df.values[0.7*m:,-1]
               
    def calweight(self):
        m,n=np.shape(self.train)
        weight=np.ones((n))
        label=self.train_label
        train=self.train
        
        for i in xrange(self.maxcycle):
            weight=weight+self.getgradient(train,weight,label).dot(self.ratio)
            
        self.weight=weight
        return weight
        
    def getgradient(self,train,weight,label):
        gradient=0    
        
        for i in range(np.shape(train)[0]):
            gradient=gradient+train[i].dot(label[i]-self.logisticfunc(train[i,:],weight))
            
        return gradient
        
    def get_stochasticgradient(self,train,weight,label):
        gradient=0    
        
        for i in range(np.shape(train)[0]):
            gradient=gradient+train[i].dot(label[i]-self.logisticfunc(train[i,:],weight))
            
        return gradient
        
    def logisticfunc(self,inX,weight):
        
        res=np.exp(inX.dot(weight))/float(1+np.exp(inX.dot(weight)))
        return res
        
LR=LogisticRegression()
path='./dataset/LRDataSet.csv'
LR.loaddataset(path)
weight=LR.calweight()

#LR.Classify(inX)

