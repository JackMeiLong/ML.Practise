# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:15:34 2016

@author: meil

Logistic Regression

http://blog.csdn.net/Fishmemory/article/details/51603836

Newton Method for Logistic Regression
"""

import numpy as np
import pandas as pd

class LogisticRegression(object):
    
    def __init__(self):
        self.train=[]
        self.train_label=[]
        self.train_df=[]
        self.weight=[]
        self.maxcycle=10
        self.test=[]
        self.test_label=[]
        self.features=[]
    
    def loaddataset(self,path):
        df=pd.read_csv(path)
        m=np.shape(df.values)[0]
        x0=pd.DataFrame([1]*m,columns=['Feature0'])
        self.train=pd.concat([x0,df],axis=1).values[:,:-1]
        self.train_label=df.values[:,-1]
               
    def calweight(self):
        m,n=np.shape(self.train)
        weight=np.zeros((n))
        label=self.train_label
        train=self.train
        
        for i in xrange(self.maxcycle):
            gradient=self.get_Gradient(train,weight,label)
            hessematrix=self.get_HesseMatrix(train,weight,label)
            weight=weight-np.linalg.pinv(hessematrix).dot(gradient)
            
        self.weight=weight
        return weight
        
    def get_Gradient(self,train,weight,label):
        m=np.shape(train)[0]
        gradient=np.zeros((np.shape(train)[1]))   
        
        for i in range(np.shape(train)[0]):
            gradient=gradient+train[i].dot(self.logisticfunc(train[i,:],weight)-label[i])
            
        gradient=gradient.dot(1/float(m))
        
        return gradient
        
    def get_HesseMatrix(self,train,weight,label):
        n=np.shape(train)[0]
        m=np.shape(weight)[0]        
        hessematrix=np.zeros((m,m))
        
        for i in range(m):
            for j in range(m):
                for t in range(m):
                    hessematrix[i][j]=hessematrix[i][j]+train[t][i]*train[t][j]*\
                    self.logisticfunc(train[t],weight)*(1-self.logisticfunc(train[t],weight))                   
        
        hessematrix=hessematrix.dot(1/float(n))
        return hessematrix
        
    def logisticfunc(self,inX,weight):
        
        if np.exp(inX.dot(weight))==np.inf:
            res=0
        else:
            res=1/float(1+np.exp(-1*inX.dot(weight)))
            
        return res
        
LR=LogisticRegression()
path='./dataset/LRDataSet2.csv'
LR.loaddataset(path)
weight=LR.calweight()

#LR.Classify(inX)

