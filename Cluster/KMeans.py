# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 22:39:41 2016

@author: jack

Notes:

K-means for clustering

Distance: Euclid
"""
from numpy import *
import math

def loadDataSet(path):
    recordlist=[]
    fp=open(path,'rb')
    content=fp.read()
    fp.close()
    rowlist = content.splitlines()
    recordlist=[row.split('\t') for row in rowlist if row.strip()]
    return recordlist
    
def getRandCents(dataset,k):
    m,n=shape(dataset)
    randcents=ones((m,k))
    
    for i in range(n):
        maxi=max(dataset[:,i])
        mini=min(dataset[:,i])
        step=float(maxi-mini)
        randcents[:,i]=mini+step*random.rand(k)
        
    return randcents

def euclid(vecA,vecB):
     vecA=mat(vecA)
     vecB=mat(vecB)
     result=math.sqrt(math.pow(vecA-vecB,2))
     return result
    
def cluster(dataset,randcents,k,maxiter=100):
    clusters=zeors((shape(dataset)[0]))
    centers=randcents
    #centers changed
    clustersteady=False
    itera=1
    m,n=shape(dataset)
    
    while (itera<maxiter or clustersteady==False):
      
        for i in xrange(m):
            dist=zeros((k,1))
            for j in xrange(k):
                dist[j]=euclid(dataset[i,:],centers[j,:])
            indices=dist.argsort()
            clusters[i]=indices[0]
        
        labels=array(list(set(cluster)))
        
        #calculate new centers
        for i in xrange(shape(labels)):
            num=0
            for j in xrange(shape(cluster)):
                if cluster[j]==labels[i]:
                    summary+=dataset[j]
                    num+=1
            
            if centers[j]==summary/num:
                clustersteady=True
            else:
                clustersteady=False
            
            centers[j]=summary/num
        
        itera+=1
        
    return cluster,centers 
        
        
            
        