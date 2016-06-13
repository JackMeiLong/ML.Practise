# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:25:00 2016

@author: meil

KNN

Input:
path,k

Output:
finalclass

Parameter:
distFun

"""
import sys
import io
from numpy import *
import math
import operator

def loaddataset(path):
    
    fp=open(path,'rb')
    content=fp.read()
    fp.close()
    rowlist=content.splitlines()
    
    dataset=[row.split('\t') for row in rowlist if row.strip()]
    
    return dataset
    
def file2matrix(dataset):
    
    dataset=array(dataset)
    m,n=shape(dataset)
    trainset=zeros((m,n-1))
    labels=ones((m,1),dtype=str)
    for i in xrange(m):
        for j in xrange(n-1):
            trainset[i,j]=float(dataset[i,j])
    
    labels=dataset[:,-1]
    
    return trainset,labels
    
def KNN(trainset,labels,testvec,k):

    m,n=shape(trainset)
    traindata=trainset
    distdic={}
    for i in xrange(m):
        dist=distEclud(traindata[i,:],testvec)
        distdic[i]=dist
    
    distlist=sorted(distdic.iteritems(),key=operator.itemgetter(1),reverse=False)
    
    classdict={}
    
    for i in xrange(k):
        label=labels[distlist[i][0]]
        classdict[label]=classdict.get(label,0)+1
    
    finalclass=sorted(classdict.iteritems(),key=operator.itemgetter(1),reverse=True)[0][0]
    
    return finalclass             
    
def distEclud(vecA,vecB):
    return linalg.norm(vecA-vecB)
    

dataset=loaddataset('testdata/knntrain.txt')
trainset,labels=file2matrix(dataset)
testvec=[14488,8.327,0.954]
k=10

finalclass=KNN(trainset,labels,testvec,k)
