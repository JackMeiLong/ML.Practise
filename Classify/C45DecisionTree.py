# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 22:12:59 2016

@author: jack
"""
#dtree=C45DecisionTree()
#dtree.loadDataSet("dataset.dat",["age","revenue","student","credit"])
#bestFeature,gainList=dtree.getBestFeature(dtree.dataset)
#dtree.train()
#testVec=['0','1','0','0']
#dtree.predict(dtree.tree,testVec)
#0.9544340029249649
#[0.2657121273840978, 0.01774123883005585, 0.17385686963473068, 0.04631324460790953]
from numpy import *
import math
import copy
import matplotlib.pyplot as plt
import treePlotter as tp 

class C45DecisionTree(object):
    
    def __init__(self):
        self.tree={}
        self.dataset=[]
        self.labels=[]
        
    def loadDataSet(self,path,labels):
        recordlist=[]
        fp=open(path,'rb')
        content=fp.read()
        fp.close()
        rowlist = content.splitlines()
        recordlist=[row.split('\t') for row in rowlist if row.strip()]
        self.dataset=recordlist
        self.labels=labels
        
    def train(self):
        labels=copy.deepcopy(self.labels)
        self.tree=self.buildTree(self.dataset,labels)
        
    def buildTree(self,dataset,labels):       
        cateList=[ dataline[-1]  for dataline in dataset]
        
        if cateList.count(cateList[0])==len(cateList):
            return cateList[0]
#        if len(dataset[0])==1:
#            return self.maxCate(cateList)
        
        
        bestFeature,gainList=self.getBestFeature(dataset)
        bestFeatLabel = labels[bestFeature]
        tree = {bestFeatLabel:{}}			
        del(labels[bestFeature])
        uniqueVals = set([data[bestFeature] for data in dataset]) 
        for value in uniqueVals:
            subLabels=labels[:]
            subDataset=self.getSubDataSet(dataset,bestFeature,value)
            subTree=self.buildTree(subDataset,subLabels)
            tree[bestFeatLabel][value] = subTree
        return tree 
        
    def calEntropy(self,dataset):
        entropy=0
        cateList=[ dataline[-1]  for dataline in dataset]
        iterms=dict([ (i,cateList.count(i) ) for i in cateList])
        datalen=len(cateList)
        for key in iterms:
            prob=float(iterms[key])/datalen
            entropy=entropy-prob*math.log(prob,2)
        return entropy
    
    def getBestFeature(self,dataset):
        maxgain=0.0
        maxratiogain=0.0
        gainList=[]
        bestFeature=-1
        baseentropy=self.calEntropy(dataset)
        for i in xrange(len(dataset[0])-1):
           uniFeatSet=set([ feat[i] for feat in dataset])
           featList=[feat[i] for feat in dataset]
           entropy=0.0
           ratio_gain=0.0
           split_info=0.0
           for val in uniFeatSet:
               prob=float(featList.count(val))/len(featList)
               subDataset=self.getSubDataSet(dataset,i,val)
               entropy+=prob*self.calEntropy(subDataset)
               split_info-=prob*math.log(prob,2)
           gain=baseentropy-entropy
           ratio_gain=float(gain)/(split_info+1e-6)
           if ratio_gain>=maxratiogain:
                maxratiogain=ratio_gain
                bestFeature=i
           gainList.append(ratio_gain)
        return bestFeature,gainList
               
    def getSubDataSet(self,dataset,feat,value):
       # featValue=[dataLine[feat] for dataLine in dataset]
        subdataset=[]
        for dataline in dataset:
            subdataline=[]
            if dataline[feat]==value:
                subdataline=dataline[:feat]
                subdataline.extend(dataline[feat+1:])
                subdataset.append(subdataline)
        return subdataset
            
    def plotDecisionTree(self,dataset,dtree):
        tp.createPlot(dtree)
        
    def predict(self,tree,testVec):
        res=''
        cateList=[dataline[-1] for dataline in self.dataset]
        root=tree.keys()[0]
        paths=tree[root].keys()
        i=0
        if testVec[i] in paths:
            index=paths.index(testVec[i])
            if tree[root][paths[index]] in cateList:
                 res=tree[root][paths[index]]
            else:
                  testVec=testVec[i+1:]
                  res=self.predict(tree[root][paths[index]],testVec)
                     
        return res