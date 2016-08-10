# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:59:31 2016

@author: meil
"""



# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 15:39:59 2016

@author: mellon

CARTDecisionTree
http://blog.csdn.net/wzmsltw/article/details/51057311
"""

import numpy as np
import pandas as pd
import math
import copy
import treePlotter as tp

class CARTDecisionTree(object):
    
    def __init__(self):
        self.train=[]
        self.train_df=[]
        self.label=[]
        self.test=[]
        self.features=[]
        self.tree={}
        self.dataset=[]
        self.validate=[]
        self.validate_label=[]
        self.validate_df=[]
    
    def loaddataset(self,path,module):
        df=pd.read_csv(path)
        
        if module=='train':
            self.features=df.columns.tolist()[:-1]
            self.train=df.values[:,:-1]
            self.train_df=df.drop(['label'],axis=1)
            self.label=df.values[:,-1]
            self.dataset=df.values
        elif module=='validate':
            self.validate_df=df.drop(['label'],axis=1)
            self.validate=df.values[:,:-1]
            self.validate_label=df.values[:,-1]   
        
    def BuildTree(self,dataset,features):
        tree={}
        features=copy.copy(features)
        bestfeature,bestindex=self.SelectBestFeature(dataset,features)
        bestvalue=set(dataset[:,bestindex])    
        bestvalue_full=set(self.train_df[bestfeature].values.tolist())
        features.append('class')
        dataset=pd.DataFrame(dataset,columns=features)
        tree[bestfeature]={}
        print 'BestFeature: {0}'.format(bestfeature)
        
        issplit=False
        
        label_majority=self.SelectBestLabel(dataset.values)               
        
        for t in bestvalue:
            subdata=dataset[dataset[bestfeature]==t].drop([bestfeature],axis=1).values
            features=dataset[dataset[bestfeature]==t].drop([bestfeature],axis=1).columns.tolist()[:-1]
            tree[bestfeature][t]=self.SelectBestLabel(subdata)
        
        issplit=self.IsSplit(label_majority,tree)
        
        if issplit==False:
            tree=label_majority
            return tree
        
        for t in bestvalue:
            bestvalue_full.remove(t)
            subdata=dataset[dataset[bestfeature]==t].drop([bestfeature],axis=1).values
            features=dataset[dataset[bestfeature]==t].drop([bestfeature],axis=1).columns.tolist()[:-1]
            if len(set(subdata[:,-1]))==1:
                tree[bestfeature][t]=subdata[0,-1]
            elif len(set(subdata[:,-1]))>1 and len(features)==0:
                tree[bestfeature][t]=self.SelectBestLabel(subdata)
            else:
                tree[bestfeature][t]=self.BuildTree(subdata,features)
        
        for t in bestvalue_full:
            tree[bestfeature][t]=self.SelectBestLabel(dataset.values)        
        
        self.tree=tree        
        return tree       
        
    def SelectBestFeature(self,dataset,features):
        gini_dict={}
        m=float(dataset.shape[0])
        data_df=pd.DataFrame(dataset)
        
        for i in features:
            gini_dict[i]=0
        
        for j in range(len(features)):
             temp=set(dataset[:,j])
             temp_dict={}
             
             for t1 in temp:
                 temp_dict[t1]=0
             for t2 in dataset[:,j]:
                 temp_dict[t2]=temp_dict[t2]+1
             
             for t3 in temp:
                 subdata=data_df[data_df[j]==t3].values
                 gini_dict[features[j]]=gini_dict[features[j]]+(self.CalculateGini(subdata))*(temp_dict[t3]/m)
            
        bestfeature=sorted(gini_dict.items(), lambda x, y: cmp(x[1], y[1]))[0][0]
        bestindex=features.index(bestfeature)
        return bestfeature,bestindex
    
    def CalculateGini(self,dataset):
        label=dataset[:,-1]
        label_uni=set(label)
        m=float(len(label))
        label_dict={}
        
        for i in label_uni:
            label_dict[i]=0
        
        for j in label:
            label_dict[j]=label_dict[j]+1
        
        gini=0
        for k,v in label_dict.iteritems():
            p=v/m
            gini=gini+math.pow(p,2)
        
        gini=1-gini
        
        return gini
        
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
        
    def IsSplit(self,label_majority,tree):
        issplit=False
        validate_data=self.validate
        validate_label=self.validate_label
        m=np.shape(validate_data)[0]
        
        accuracy_before=list([label_majority]*m==validate_label).count(True)/float(m)
        accuracy_after=self.GetAccuracy(validate_data,validate_label,tree)
        
        if accuracy_after>accuracy_before:
            issplit=True
            
        return issplit
        
    def GetAccuracy(self,data,label_true,tree):
        accuracy=float(0)
        m=np.shape(data)[0]
        label_pred=['']*m
        
        for i in range(m):
            label_pred[i]=self.predict(data[i,:],tree)
        accuracy=list(label_pred==label_true).count(True)/float(m)
        
        return accuracy
        
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
            
path='./dataset/watermellontrain.csv'
path_validate='./dataset/watermellonvalidate.csv'
cart=CARTDecisionTree()
cart.loaddataset(path,'train')
cart.loaddataset(path_validate,'validate')
tree=cart.BuildTree(cart.dataset,cart.features)

path_test='./dataset/watermellonvalidate.csv'
inX=pd.read_csv(path_test).values
y_pred=['']*np.shape(inX)[0]

for i in range(np.shape(inX)[0]):
    y_pred[i]=cart.predict(inX[i],tree)

accuracy=list(y_pred==cart.validate_label).count(True)/float(np.shape(inX)[0])