# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:51:22 2016

@author: meil
Kmeans

"""

import numpy as np
import pandas as pd
from enum import enum

class KMeans(object):
    
    def __init__(self):
        self.dataset=[]
        self.randomcent=[]
        self.cent=[]
        self.label=[]
        self.iteration=50
        
    def import_data(self,path):
        data_origin=pd.read_csv(path)
        self.dataset=data_origin.valus
        
    def init_cent(self,k):
        data=self.dataset
        m,n=np.shape(data)
        random_cent=np.zeros((k,n))
        
        for i in range(k):
            min_value=np.min(data,axis=0)
            max_value=np.max(data,axis=0)
            step=max_value-min_value
            random_cent[i]=min_value+step*np.random.rand()
        
        self.randomcent=random_cent
        return random_cent
    
    def distance(self,vecA,vecB,mode,p):
        
        dist=0
        vecA=np.array(vecA)
        vecB=np.array(vecB)
        diff=np.abs(vecA-vecB)
        if mode=='Mink':
            dist=np.sum(np.power(diff,p))**(1./p)
        elif mode=='Euclid':
            dist=np.sum(np.power(diff,2))**(1./2)
        elif mode=='Manh':
            dist=np.sum(diff)
        elif mode=='Cheby':
            dist=np.max(diff)
            
        return dist
    
    def cluster(self,k):
        flag_changed=True
        iteration=0
        data=self.dataset
        m,n=np.shape(data)       
        randomcent=self.randomcent
        labels=np.linspace(1,k,k)
        data_labels=np.zeros((m,1),dtype='str')
        centers=randomcent
        
        while (iteration<=self.iteration & flag_changed==True):
            
            flag_changed=True
            
            for i in range(m):
                dist_min=self.distance(data[i,:],randomcent[0,:],mode='Eculid')
                data_labels[i]='1'
                for j in range(k):
                    dist=self.distance(data[i,:],randomcent[j,:],mode='Eculid')
                    if dist<=dist_min:
                        data_labels[i]=str(j+1)
            
            labels=np.array(labels)
            
            for i in labels:
                sum_cent=np.zeros((1,n))
                num=0               
                for j in range(m):
                    if data_labels[j]==i:
                        sum_cent=sum_cent+data[j,:]
                        num=num+1
                if centers[i-1]==sum_cent/float(num):
                   flag_changed=flag_changed & False
              
                centers[i-1]=sum_cent/float(num)             
          
            iteration=iteration+1
          
               

Mode=enum.Enum('Mode','Mink Euclid Manh Cheby')