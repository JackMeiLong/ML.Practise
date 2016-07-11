# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:04:35 2016

@author: meil

self.hn: number of hidden neurons
self.hc: centers of hidden neurons
self.hv：variance of hidden neurons

Gauss function

Calculate weight between hidden-layer and output-layer
LMS: Least Mean Square

"""
from numpy import *

class RBFNN(object):
    
    def __init__(self):
        self.trainset=[]
        self.trainlabel=[]
        self.weight=[]
        self.hnn=100
        self.hc=[]
        self.hv=[]
        
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
        
    def euclid(vecA,vecB):
        vecA=mat(vecA)
        vecB=mat(vecB)
        result=math.sqrt(math.pow(vecA-vecB,2))
        return result
    
    def manhattan(vecA,vecB):
        vecA=mat(vecA)
        vecB=mat(vecB)
        result=sum(abs(vecA-vecB))
        return result
        
    def guass(self,i,j):
        result=math.exp(-1*math.pow(self.trainset[i]-self.hc[j],2)/(2*self.hv[j]))
        return result
    
    #init the number ,centers and variance of hidden neurons    
    def initNN(self,num=shape(self.trainset)[0],dis=euclid):
        self.hnn=num
        self.hc=self.trainset
        self.hv=zeros((shape(self.trainset)[0]))
        
        for i in xrange(self.hnn):
            maxdis=0
            maxind=0
            for j in xrange(self.hnn):
                if dis(self.trainset[i],self.trainset[maxind])>=maxdis:
                    maxdis=dis(self.trainset[i],self.trainset[maxind])
                    maxind=j
            self.hv[i]=float(maxdis/math.sqrt(2*self.hnn))            
    
    def calWeight(self):
        weight=zeors((shape(self.trainset)[1]))
        phi=zeros((shape(self.trainset)[0],self.hnn))
        m,n=shape(phi)
        
        for i in xrange(m):
            for j in xrange(n):
                phi[i][j]=self.guass(i,j)
        
        phit=mat(phi).T
        
        if linalg.det(mat(phit)*mat(phi))!=0:
            weight=linalg.det(mat(phit)*mat(phi))*phit*mat(self.trainlabel)
        else:
            print 'matrix is not singular'
         
        self.weight=weight
    
    def predict(self,inX):
        outY=mat(inX).T*self.weight
        return outY


rbf=RBFNN()
path='testSet.txt'
dataset=rbf.loadtrainset(path)
rbf.file2matrix(dataset)
rbf.initNN()
rbf.calWeight()
#input inX
rbf.predict(inX)
      
        
    
    