# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:09:26 2016

@author: meil

LinearRegression

testVec: coloumn vector

"""

from numpy import *
import math
import sys
import os

class LinearRegression(object):
    
    def __init__(self):
        self.trainset=[]
        self.trainlabel=[]
        self.weight=[]
    
    def loadtrainset(self,path):
        fp=open(path,'rb')
        content=fp.read()
        fp.close()
        rowlist=content.splitlines()
        dataset=[row.split('\t') for row in rowlist if row.strip()]
        return dataset
        
        
    def file2matrix(self,dataset):
        m,n=shape(dataset)
        self.trainset=zeros((m,n-1))
        self.trainlabel=zeros((m,1))
        
        
        self.trainset=dataset[:,:n-1]
        self.trainlabel=dataset[:,-1]
    
    def calweight(self):
        m,n=shape(self.trainset)
        weight=zeros((n,1))
        trainX=mat(self.trainset)
        trainY=mat(self.trainlabel)
        xTx=trainX.T*trainX
        
        if linalg.det(xTx)!=0:
            weight=linalg.inv(xTx)*trainX.T*trainY
        self.weight=weight
    
    def predict(self,testVec):
        pY=mat(testVec).T*self.weight
        return pY


LR=LinearRegression()
path='testSet.txt'
dataset=LR.loadtrainset(path)
LR.file2matrix(dataset)
LR.calweight()
#input inX
outY=LR.predict(inX)


