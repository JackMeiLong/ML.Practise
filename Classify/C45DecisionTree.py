# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 10:08:09 2016

@author: mellon
C4.5 
"""

import numpy as np
import pandas as pd
import math
import copy

class C45DecisionTree(object):
    
    def __init__(self):
        self.train=[]
        self.label=[]
        self.test=[]
        self.features=[]
        self.tree={}
        self.dataset=[]
    
    def loaddataset(self,path):
        df=pd.read_csv(path,dtype={'windy':np.str})
        self.features=df.columns.tolist()[:-1]
        self.train=df.values[:,:-1]
        self.label=df.values[:,-1]
        self.dataset=df.values
      
    def BuildTree(self,dataset,features):
        tree={}
        features=copy.copy(features)
        bestfeature,bestindex=self.SelectBestFeature(dataset,features)
        bestvalue=set(dataset[:,bestindex])       
        features.append('class')
        dataset=pd.DataFrame(dataset,columns=features)
        tree[bestfeature]={}
        print 'BestFeature: {0}'.format(bestfeature)
        
        for t in bestvalue:
            subdata=dataset[dataset[bestfeature]==t].drop([bestfeature],axis=1).values
            features=dataset[dataset[bestfeature]==t].drop([bestfeature],axis=1).columns.tolist()[:-1]
            if len(set(subdata[:,-1]))==1:
                tree[bestfeature][t]=subdata[0,-1]
            elif len(set(subdata[:,-1]))>1 and len(features)==0:
                tree[bestfeature][t]=self.SelectBestLabel(subdata)
            else:
                tree[bestfeature][t]=self.BuildTree(subdata,features)
        self.tree=tree        
        return tree       
        
    def SelectBestFeature(self,dataset,features):
        info=self.CalcuteEntropy(dataset)
        info_dict={}
        split_info={}
        m=float(dataset.shape[0])
        data_df=pd.DataFrame(dataset)
        
        for i in features:
            info_dict[i]=0
            split_info[i]=0
        
        for j in range(len(features)):
             temp=set(dataset[:,j])
             temp_dict={}
             
             for t1 in temp:
                 temp_dict[t1]=0
             for t2 in dataset[:,j]:
                 temp_dict[t2]=temp_dict[t2]+1
             
             for t3 in temp:
                 subdata=data_df[data_df[j]==t3].values
                 info_dict[features[j]]=info_dict[features[j]]+(self.CalcuteEntropy(subdata))*(temp_dict[t3]/m)
                 split_info[features[j]]=split_info[features[j]]-(temp_dict[t3]/m)*math.log(temp_dict[t3]/m)
            
             info_dict[features[j]]=(info-info_dict[features[j]])/float(split_info[features[j]])
        
        bestfeature=sorted(info_dict.items(), lambda x, y: cmp(x[1], y[1]))[-1][0]
        bestindex=features.index(bestfeature)
        return bestfeature,bestindex
    
    def CalcuteEntropy(self,dataset):
        label=dataset[:,-1]
        label_uni=set(label)
        m=float(len(label))
        label_dict={}
        
        for i in label_uni:
            label_dict[i]=0
        
        for j in label:
            label_dict[j]=label_dict[j]+1
        
        entropy=0
        for k,v in label_dict.iteritems():
            p=v/m
            entropy=entropy+p*math.log(p)
        
        entropy=-1*entropy
        return entropy
        
    def SelectBestLabel(self,subdata):
        tarray=subdata[:,-1]
        tset=set(tarray)
        tdict={}
        
        for i in tset:
            tdict[i]=0
        
        for i in tarray:
            tdict[i]=tdict[i]+1
        
        tbest=sorted(tdict.items(),lambda x,y:cmp(x[1],y[1]))[-1][0]
        return tbest
        
    def predict(self,inX,tree):
        root=tree.keys()[0]
        paths=tree[root].keys()
        labels=self.label.tolist()
        features=copy.copy(self.features)
        res=''
        ix=features.index(root)
        for i in paths:
            if inX[ix]==i:
                if tree[root][i] in labels:
                    res=tree[root][i]
                else:
                    tree=tree[root][i]
                    res=self.predict(inX,tree)
        return res  
            
path='./dataset/c45train.csv'
c45=C45DecisionTree()
c45.loaddataset(path)
tree=c45.BuildTree(c45.dataset,c45.features)
path_test='./dataset/c45test.csv'
inX=pd.read_csv(path_test,dtype = {'windy': np.str}).values
y_pred=np.zeros((np.shape(inX)[0],1),dtype='str')

for i in range(np.shape(inX)[0]):
    y_pred[i]=c45.predict(inX[i],tree)

