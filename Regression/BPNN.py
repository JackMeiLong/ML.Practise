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
        self.n=5
        self.l=10
        self.p=1
        self.wij=[]
        self.wjk=[]
        self.aj=[]
        self.bk=[]
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
        self.p=1
        self.wij=np.ones((self.n,self.l))
        self.wjk=np.ones((self.l,self.p))
    
    def train(self):
        train_set=self.train_set
        train_label=self.train_label
        #wjk
        
        #wij
        
        #aj
        
        #bk
        
    def calHj(self):
        
    def calOk(self):
    
    def logistic(self,inx):
        res=np.exp(inx)/float(1+np.exp(inx))
        return res
        
