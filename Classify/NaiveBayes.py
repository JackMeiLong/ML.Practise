# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 18:55:51 2016

@author: meil

Naive Bayes

教育,政治上的反复无常,的话题,据当地媒体营业额和地方,政治和社会空间

"""

import numpy as np
import math
import pandas as pd

class NaiveBayes(object):
    
    def __init__(self):
        self.train=[]
        self.train_df=[]
        self.train_label=[]
        self.features=[]
        self.test=[]
        self.test_label=[]
    
    def loaddata(self,path):
        df=pd.read_csv(path)
        self.features=df.columns
        m=np.shape(df.values)[0]
        self.train=df.values[:0.7*m,:-1]
        self.train_label=df.values[:0.7*m,-1]
        self.train_df=df
        self.test=df.values[0.7*m:,:-1]
        self.test_label=df.values[0.7*m:,-1]
    
    def classify(self,inX):
        train_df=self.train_df
        features=self.features
        label=self.train_label
        label_set=set(label)
        label_dict={}
        
        for i in label_set:
            label_dict[i]=0
        
        for i in label:
            label_dict[i]=label_dict[i]+1
        
        for k,v in label_dict.iteritems():
            label_dict[k]=v/float(np.shape(label)[0])  
        
        for i in label_set:
            num=1            
            temp=train_df[train_df[feaures[-1]]==i].values
            num_label=np.shape(temp)[0]
            num_value=0.1
            for j in range(np.shape(features)[0]-1):
                temp1=temp[temp[features[features[j]]==inX[j]].values
                num_value=np.shape(temp1)[0]
                prob=num_value/num_label
                num=num*prob
            
            label_dict[i]=label_dict[i]*num
        
        bestlabel=sorted(label_dict.items,lambda x,y:cmp(x[1],y[1]))[-1][0]
        return bestlabel
    
    def predict(self,inX):
        
        label=self.classify(inX)
        return label

path='./dataset/blogger.csv'
nb=NaiveBayes()
nb.loaddata(path)
inX=nb.test[0,:]

y_pred=nb.predict(inX)

        