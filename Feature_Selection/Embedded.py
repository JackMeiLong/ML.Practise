# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:25:49 2016

@author: meil
"""
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier

class Embedded(object):
    
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    
    def feature_embedded(self,estimator):
        X=self.X
        Y=self.Y
        sel=SelectFromModel(estimator)
        res=sel.fit_transform(X, Y)
        return sel,res
    
iris=load_iris()

embedded=Embedded(iris.data,iris.target)
    
#penalty
LR_0=LogisticRegression(penalty='l1')
sel_0,res_0=embedded.feature_embedded(LR_0)
