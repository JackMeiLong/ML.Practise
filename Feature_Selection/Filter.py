# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:14:27 2016

@author: meil
"""
import numpy as np

from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
#from minepy.minepy import MINE

from sklearn.datasets import load_iris

class Filter(object):
    
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    
    def Variance(self,k):
        X=self.X
        sel=VarianceThreshold(threshold=k)
        res=sel.fit_transform(X)
        return res
        
    def Pearson(self,k):
        X=self.X
        Y=self.Y
        pearson=lambda X,Y:np.array(map(lambda x:pearsonr(x,Y),X.T)).T
        sel=SelectKBest(pearson,k)
        res=sel.fit_transform(X,Y)
        return res
    
    def mic(x,y):
        m=MINE()
        m.compute_score(x,y)
        return (m.mic(),0.5)
        
    def Mic(self,k):
        X=self.X
        Y=self.Y
        Mic=lambda X,Y:np.array(map(lambda x:self.mic(x,Y),X.T)).T
        sel=SelectKBest(Mic,k)
        res=sel.fit_transform(X,Y)
        return res

iris=load_iris()

fil=Filter(iris.data,iris.target)

k_0=0.5
res_0=fil.Variance(k_0)

k_1=2
res_1=fil.Pearson(k_1)