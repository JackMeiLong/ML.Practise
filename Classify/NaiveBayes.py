# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 18:55:51 2016

@author: meil

Naive Bayes

"""

import numpy as np

class NaiveBayes(object):
    
    def __init__(self):
        self.trainset=[]
        self.trainlabel=[]
        self.testset=[]
        self.testlabel=[]
    
    def classify(self):
        trainset=self.trainset
        trainlabel=self.trainlabel
        
        feature_n=np.shape(trainset)[1]
        
        