# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 22:06:53 2016

@author: mellon
BPNN

Αα alpha a:lf 阿尔法 2 Ββ beta bet 贝塔 3 Γγ gamma ga:m 伽马 4 Δδ delta delt 德尔塔 
http://blog.csdn.net/google19890102/article/details/32723459
"""
import numpy as np
import pandas as pd

class BPNN(object):
    
    def __init__(self):
        self.train_set=[]
        self.train_label=[]
        self.n=5#number of neurons of input layer
        self.l=10#number of neurons of hidden layer
        self.p=1#number of neurons of output layer
        self.wij=[]#weight of input & hidden
        self.wjk=[]#weight of hidden & output
        self.aj=[]#threshold of hidden
        self.bk=[]#threshold of output
        self.alpha=0.3
        self.beta=0.5
        self.gamma=0.3
        self.delta=0.5
    
    def loaddata(self,path):
        df=pd.read_csv(path)
        self.train_set=df.values[:,:-1]
        self.train_label=df.values[:,-1]
        
    def init_parameter(self,hidden):        
        self.n=np.shape(self.train_set)[0]
        self.l=hidden
        self.m=1
        self.wij=np.ones((self.n,self.l))
        self.wjk=np.ones((self.l,self.p))
        self.aj=np.ones((self.l,1))
        self.bk=np.ones((self.m,1))
    
    def train(self):
        train_set=self.train_set
        train_label=self.train_label
        num=np.shape(train_set)[0]
        wij=self.wij
        wjk=self.jk
        aj=self.aj
        bk=self.bk
        n=self.n
        l=self.l
        m=self.m
        for t in xrange(num):
            #wjk            
            for j in xrange(l):
                for k in xrange(m):
                    Hj=wij[:,j].dot(train_set[t,:])+aj[j]
                    Hj=self.logistic(Hj)
                    wjk[j,k]=wjk[j,k]+self.alpha*train_label[t]*Hj
            
            #wij
            for i in xrange(n):
                for j in xrange(l):
                    for k in xrange(m):
                        Hj=wij[:,j].dot(train_set[t,:])+aj[j]
                        Hj=self.logistic(Hj)
                        ok=wjk[:,k].dot(Hj)+bk[k]
                    
                    wij[i,j]=wij[i,j]+self.beta*((train_label[t]-ok).dot(wjk[j,:]))*self.differential(Hj)*train_set[t,i]
            
            #bk
            for k in xrange(m):  
                for j in xrange(l):
                    Hj=wij[:,j].dot(train_set[t,:])+aj[j]
                    Hj=self.logistic(Hj)
                ok=wjk[:,k].dot(Hj)+bk[k]
                bk[k]=bk[k]+train_label[t].dot(ok)*self.gamma
        
            #aj
            for j in xrange(l):
               Hj=wij[:,j].dot(train_set[t,:])+aj[j]
               Hj=self.logistic(Hj)
               for k in xrange(m):
                   ok=wjk[:,k].dot(Hj)+bk[k]
               aj[j]=aj[j]+self.delta*((train_label[t]-ok).dot(wjk[j,:]))*self.differential(Hj)
    
    def logistic(self,inx):
        res=np.exp(inx)/float(1+np.exp(inx))
        return res
     
    def differential(self,inx):
        res=(1-self.logistic(inx))*self.logistic(inx)
        return res

bpnn=BPNN()
path='./dataset/student.csv'
bpnn.loaddata(path)
bpnn.train()

