# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:49:21 2016

@author: meil
"""
from numpy import *
import matplotlib.pyplot as plt

class RidgeRegression(object):
    
    def __init__(self):
        self.trainset=[]
        self.trainlabel=[]
        self.weight=[]
        self.k=range(5)
        
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
    
    def calweight(self,k):
        m,n=shape(self.trainset)
        xT=mat(self.trainset).T
        xTx=xT*mat(self.trainset)
        weight=zeros((n,1))
        
        if linalg.det(xTx)==0:
            #ridge regression
            weight=linalg.inv(xTx+k*eye(n))*xT*mat(self.trainlabel)
        else:
            #linear regression
            weight=linalg.inv(xTx)*xT*mat(self.trainlabel)

        self.weight=weight
        
        
    def plottrace(self):
        #plot ridge trace
        weight=zeors((shape(self.k)[0]))
        for k in self.k:
            weight[k]=self.calweight(k)
            plt.plot(weight[k],linestyle='-',label='ridge trace & k={0}'.format(k))
        
        plt.show()
        
    def predict(self,inX):
        outY=mat(inX).T*mat(self.weight)
        return outY

RR=RidgeRegression()
path='testSet.txt'
dataset=RR.loaddataset(path)

RR.file2matrix(dataset)

RR.plottrace()

#inX
outY=RR.predict(inX)

