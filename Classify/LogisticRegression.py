# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:15:34 2016

@author: meil

Logistic Regression



"""

from numpy import *
import math
import sys
import io

class LogisticRegression(object):
    
    #Constructor
    def __init__(self):
        self.trainset=[]
        self.trainlabel=[]
        self.weight=[]
        self.ratio=0.01
        self.maxcycle=10
    
    #Load TrainData
    def loaddataset(self,path):
        fp=open(path,'rb')
        content=fp.read()
        fp.close()
        rowlist=content.splitlines()
        dataset=[row.split('\t') for row in rowlist if row.strip()]
        dataset=mat(dataset)
        return dataset
    #file to matrix
    def file2matrix(self,dataset):
        m,n=shape(dataset)
        trainset=zeros((m,n-1))
        trainlabel=zeros((m,1))
        for i in xrange(m):
            for j in xrange(n-1):
                trainset[i][j]=float(dataset[i][j])
            trainlabel[i][0]=float(dataset[i][n-1])
        self.trainset=trainset
        self.trainlabel=trainlabel
        
    #Sigmod Function
    def Sigmod(self,inX,weight):
        result=1/(1+exp(-1*inX*weight))
        return result
        
    #calculate weight
    def calweight(self):
        m,n=shape(self.trainset)
        weight=ones((n,1))
        
        for i in xrange(self.maxcycle):
            weight=weight+self.ratio*mat(self.trainset).T*(self.trainlabel-self.Sigmod(self.trainset,weight))
            
        self.weight=weight
    
    #calssify inX : column vector
    def Classify(self,inX):
        if shape(inX)[0]==shape(self.weight)[0]:
            outY=mat(inX).T*self.weight
        else:
            print 'the dimension of inX is not correct'


LR=LogisticRegression()
path='testSet.txt'
dataset=LR.loaddataset()
LR.file2matrix(dataset)
LR.calweight()

#LR.Classify(inX)

