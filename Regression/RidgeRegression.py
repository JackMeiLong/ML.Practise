# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:49:21 2016

@author: meil
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RidgeRegression(object):
    
    def __init__(self):
        self.trainset=[]
        self.trainlabel=[]
        self.weight=[]
        self.k=[]
        
    def loaddataset(self,path):
        df=pd.read_csv(path)
        self.trainset=df.values[:,:-1]
        self.trainlabel=df.values[:,-1]
    
    def calweight(self,k):
        m,n=np.shape(self.trainset)
        X=self.trainset
        Y=self.trainlabel        
        XT=np.transpose(X)
        XTX=XT.dot(X)
        weight=np.zeros((n,1))
        
        if np.linalg.det(XTX)==0:
            weight=np.linalg.inv(XTX+np.eye(n).dot(k)).dot(XT).dot(Y)
        else:
            weight=np.linalg.inv(XTX).dot(XT).dot(Y)

        self.weight=weight
        return weight
        
    def predict(self,inX):
        outY=np.transpose(inX).dot(self.weight)
        return outY

RR=RidgeRegression()
path='testSet.txt'
RR.loaddataset(path)
k=np.linspace(1,10,4)

weight=RR.calweight(k[0])


#RR.plottrace()
#
##inX
#outY=RR.predict(inX)

