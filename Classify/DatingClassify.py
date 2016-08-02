# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 10:13:55 2016

@author: meil
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.patches as mpatches
import matplotlib.pylab as plt

from sklearn.metrics import classification_report

class DatingClassify(object):
    
    def __init__(self):
        self.train=[]
        self.trainlabel=[]
        self.test=[]
        self.testlabel=[]
        self.train_scaled=[]
        self.test_scaled=[]
        self.percent=0.8
        self.df=[]
        self.features={}
    
    def loaddata(self,path):
        df=pd.read_csv(path)
        self.df=df
        data=df.values
        m,n=data.shape
        self.train=data[:m*self.percent,:-1]
        self.test=data[m*self.percent:,:-1]        
        labels=df.values[:,-1]
        le=LabelEncoder()
        labels=le.fit_transform(labels)
        self.trainlabel=labels[:m*self.percent]
        self.testlabel=labels[m*self.percent:]
        self.features[self.df.columns[0]]='Number of frequent flyer miles earned per year'
        self.features[self.df.columns[1]]='Percentage of time spent playing video games'
        self.features[self.df.columns[2]]='Liters of ice cream consumed weekly'
           
    def preprocess(self):
        train=self.train
        test=self.test
        
        minmaxscaler=MinMaxScaler(feature_range=(0,1))
        self.train_scaled=minmaxscaler.fit_transform(train)
        self.test_scaled=minmaxscaler.fit_transform(test)
    
    def KNN(self,k):
        knn=KNeighborsClassifier(n_neighbors=k,algorithm='kd_tree',p=2)
        X=self.train_scaled
        Y=self.trainlabel
        knn.fit(X,Y)
        return knn
    
    def predict(self,estimator,test):
        testlabel=estimator.predict(test)
        return testlabel
        
    def visual(self):
        plt.figure()
        plt.subplot(3,1,1)
        self.df.Feature0.plot(title=self.features['Feature0'])
        plt.subplot(3,1,2)
        self.df.Feature1.plot(title=self.features['Feature1'])
        plt.subplot(3,1,3)
        self.df.Feature2.plot(title=self.features['Feature2'])
        
        plt.figure()
        self.df.Category.value_counts().plot(kind='bar')
        
        plt.figure()
        
        colours=[]
        dic={'largeDoses':'r','smallDoses':'b','didntLike':'g'}
        self.df.Category.map(lambda x:colours.append(dic[x]))
        
        plt.scatter(self.df.Feature1.values,self.df.Feature2.values,c=colours)
        plt.xlabel(self.features['Feature1']) 
        plt.ylabel(self.features['Feature2']) 
        recs = []
        classes=['largeDoses','smallDoses','didntLike']
        color=['r','b','g']
        for i in range(len(color)):
            recs.append(mpatches.Rectangle((0,0),1,1,fc=color[i]))
        
        plt.legend(recs,classes,loc=4)
        
    def evaluate(self,testlabel):
        eva=self.testlabel==testlabel
        target_names = ['didntLike', 'largeDoses','smallDoses']
        print(classification_report(self.testlabel, testlabel, target_names=target_names))
        print 'Precision: {0}'.format(len(eva[eva==True])/float(len(testlabel)))
    
path='./dataset/datingset.csv'
dc=DatingClassify()
print 'preprocessing data'
dc.loaddata(path)
dc.preprocess()
print 'preprocessing data done'
k=1
knn=dc.KNN(k)
test_scaled=dc.test_scaled
testlabel=dc.predict(knn,test_scaled)
#dc.visual()
dc.evaluate(testlabel)

