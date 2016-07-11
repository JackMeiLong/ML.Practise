# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 16:23:40 2016

@author: meil

Practise Plot

"""

import matplotlib.pyplot as plt
from numpy import * 

def loaddataset(path):
    
    fp=open(path,'rb')
    content=fp.read()
    fp.close()
    rowlist=content.splitlines()
    dataset=[row.split('\t') for row in rowlist if row.strip()]
    dataset=mat(dataset)
    
    return dataset
    
def file2matrix(dataset):
    
    m,n=shape(dataset)
    trainset=zeros((m,n-1))
    trainlabel=zeros((m,1))
    for i in xrange(m):
        for j in xrange(n-1):  
           trainset[i][j]=float(dataset[i][j])
           trainlabel[i][0]=float(dataset[i][n-1])
    
    return trainset,trainlabel

path='testdata'

dataset=loaddataset(path)
trainset,trainlabel=file2matrix(dataset)

X=trainset[:,0]
Y=trainlabel

plt.plot(X,Y,'r+')

plt.show()
        
        
        
        