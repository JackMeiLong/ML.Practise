# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:59:55 2016

@author: meil

"""
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.datasets import load_iris

class Wrapper(object):
    
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    
    def RFE(self,estimator,k):
        X=self.X
        Y=self.Y
        rfe=RFE(estimator,n_features_to_select=k)
        res=rfe.fit_transform(X,Y)
        return rfe,res

iris=load_iris()

wrapper=Wrapper(iris.data,iris.target)
LR=LogisticRegression()
k=3
rfe,res=wrapper.RFE(LR,3)
