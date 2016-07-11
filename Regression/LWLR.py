# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:15:53 2016

@author: meil

Locally Weight Linear Regression

局部加权线性回归


"""
from numpy import *

import math

class LWLR(object):
    
    def __init__(self):
        self.trainset=[]
        self.trainlabel=[]
        self.weight=[]
        
    def loaddataset(self,path):
        fb=open(path,'rb')
        content=fp.read()
        fb.close()
        rowlist=content.splitlines()
        dataset=[row.split('\t') for row in rowlist if row.strip()]
        dataset=mat(dataset)
        return dataset
    
    def file2matrix(self,dataset):
        m,n=shape(dataset)
        self.trainset=zeors((m,n-1))
        self.trainlabel=zeros((m,1))
        
        self.loaddataset=float(dataset[:,:n-1])
        self.trainlabel=float(dataset[:,-1])
    
    def manhattan(self,vecA,trainset):
        #vecA vecB column vector
        result=sum(abs(vecA-vecB))
        return result
        
    #testpoint inX
    def Calweight(self,inX,k):
        m,n=shape(self.trainset)
        x=mat(self.loaddataset)
        theta=eye(m)
        xT=x.T
        weight=zeros((m,1))
        
        for i in xrange(m):
            diff=self.manhattan(inX,self.trainset[i,:])
            theta[i][i]=math.exp(diff/(-1*2*math.pow(k,2)))
        
        weight=linalg.inv(xT*theta*x)*xT*theta*self.trainlabel
        
        self.weight=weight
        
        outY=mat(inX).T*self.weight
         
        return outY
       
    

lwlr=LWLR()
path='testSet.txt'
dataset=lwlr.loaddataset(path)
lwlr.file2matrix(dataset)
#input inX & k
lwlr.Calweight(inX,k)